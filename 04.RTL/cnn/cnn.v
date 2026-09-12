/*
 * Module: cnn
 * The CNN classification path, presented with the port list top_wrapper
 * expects: sample blocks in, class out, valid/ready handshake.
 *
 * Chain, per classification:
 *   block_feeder     64-sample block -> serial stream, tagged by channel
 *   decimator_x64    /64, one instance per kept channel
 *   prescale         recovers the headroom decimation gives up
 *   spectrogram      32 columns x 32 bins, one per channel
 *   cnn_core         normalize, conv/ReLU/pool x8, dense, argmax
 *
 * Three things about the boundary with the rest of the design:
 *
 * Channels. The transmitter sends x_A, y_A, x_B, y_B in that order, so
 * the frame buffer's tags 0 and 1 are the mancal A pair this network was
 * trained on. Tags 2 and 3 are consumed and discarded -- `ready_o` must
 * stay asserted for them or ingestion stalls and the buffer overflows.
 * Note that ml_pipeline_fsm's localparams name tag 1 CH_X_B, which
 * contradicts the transmitter; the wire order is what is followed here.
 *
 * Class numbering. This network orders classes normal, unbalance,
 * misalignment, wear, while the LED decode in top_wrapper expects
 * normal, misalignment, unbalance, wear. Classes 1 and 2 are swapped on
 * the way out so nothing downstream has to change.
 *
 * Rate. A classification needs 576 decimated samples per channel, so
 * 36,864 raw ones, delivered 64 at a time. Everything here is idle
 * almost all of that time; the UART is slower than the datapath by
 * orders of magnitude.
 * Status: implemented.
 */
module cnn #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    /* $readmemh resolves these against the tool's working directory, so
     * the defaults assume it is the project root. A testbench run from
     * its own directory overrides them. */
    parameter KERNEL_FILE = "04.RTL/cnn/weights/conv1_kernels.hex",
    parameter CBIAS_FILE  = "04.RTL/cnn/weights/conv1_bias.hex",
    parameter DBIAS_FILE  = "04.RTL/cnn/weights/dense_b.hex",
    parameter DW0_FILE    = "04.RTL/cnn/weights/dense_w_c0.hex",
    parameter DW1_FILE    = "04.RTL/cnn/weights/dense_w_c1.hex",
    parameter DW2_FILE    = "04.RTL/cnn/weights/dense_w_c2.hex",
    parameter DW3_FILE    = "04.RTL/cnn/weights/dense_w_c3.hex"
) (
    input  wire clk,
    input  wire rst_n,

    input  wire signed [N*DATA_WIDTH-1:0] sample_block_i,
    input  wire [1:0]                     channel_i,
    input  wire                           valid_i,
    output wire                           ready_o,

    output reg                            valid_o,
    output reg  [1:0]                     class_o
);

localparam WORD_BITS  = DATA_WIDTH;
localparam SPEC_ADDR_BITS = 10;
localparam CORE_ADDR_BITS = 12;
localparam PRESCALE_SHIFT = 7;

localparam CH_X_A = 2'd0;  // wire order from the transmitter, not the
localparam CH_Y_A = 2'd1;  // localparam names in ml_pipeline_fsm

wire                        sample_valid;
wire signed [WORD_BITS-1:0] sample;
wire [1:0]                  sample_channel;

wire dec0_busy, dec1_busy;
wire spec0_col_busy, spec1_col_busy;

/* Hold off the stream while anything downstream is mid-computation: the
 * decimator must not shift its delay line during a dot product, and the
 * spectrogram must not shift its window while a column is being read
 * into the FFT. Over UART these blocks are idle almost all the time, so
 * this costs nothing -- but relying on the source being slow would make
 * correctness a property of the baud rate. */
wire stream_stall = dec0_busy || dec1_busy || spec0_col_busy || spec1_col_busy;

block_feeder #(
    .WORD_BITS(WORD_BITS),
    .N(N)
) u_feeder (
    .clk(clk),
    .rst_n(rst_n),
    .sample_block_i(sample_block_i),
    .channel_i(channel_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .dst_busy(stream_stall),
    .sample_valid_o(sample_valid),
    .sample_o(sample),
    .channel_o(sample_channel)
);

/* ---- channel 0: x_A ---- */
wire dec0_valid;
wire signed [WORD_BITS-1:0] dec0_sample, pre0_sample;

decimator_x64 #(.WORD_BITS(WORD_BITS)) u_dec0 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid && (sample_channel == CH_X_A)),
    .sample_in(sample),
    .busy(dec0_busy),
    .sample_out(dec0_sample),
    .sample_out_valid(dec0_valid)
);

prescale #(.WORD_BITS(WORD_BITS), .SHIFT(PRESCALE_SHIFT)) u_pre0 (
    .sample_in(dec0_sample),
    .sample_out(pre0_sample)
);

/* ---- channel 1: y_A ---- */
wire dec1_valid;
wire signed [WORD_BITS-1:0] dec1_sample, pre1_sample;

decimator_x64 #(.WORD_BITS(WORD_BITS)) u_dec1 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid && (sample_channel == CH_Y_A)),
    .sample_in(sample),
    .busy(dec1_busy),
    .sample_out(dec1_sample),
    .sample_out_valid(dec1_valid)
);

prescale #(.WORD_BITS(WORD_BITS), .SHIFT(PRESCALE_SHIFT)) u_pre1 (
    .sample_in(dec1_sample),
    .sample_out(pre1_sample)
);

/* ---- spectrograms ---- */
wire [SPEC_ADDR_BITS-1:0] spec_read_addr;
wire [WORD_BITS-1:0] spec0_mag, spec1_mag;
wire spec0_busy, spec1_busy, spec0_done, spec1_done;

wire spec_start, clear_done, core_start, emit;

spectrogram #(.WORD_BITS(WORD_BITS)) u_spec0 (
    .clk(clk),
    .rst_n(rst_n),
    .start(spec_start),
    .sample_valid(dec0_valid),
    .sample_in(pre0_sample),
    .busy(spec0_busy),
    .col_busy(spec0_col_busy),
    .done(spec0_done),
    .read_addr(spec_read_addr),
    .read_mag(spec0_mag)
);

spectrogram #(.WORD_BITS(WORD_BITS)) u_spec1 (
    .clk(clk),
    .rst_n(rst_n),
    .start(spec_start),
    .sample_valid(dec1_valid),
    .sample_in(pre1_sample),
    .busy(spec1_busy),
    .col_busy(spec1_col_busy),
    .done(spec1_done),
    .read_addr(spec_read_addr),
    .read_mag(spec1_mag)
);

/* ---- classifier ---- */
wire [CORE_ADDR_BITS-1:0] core_spec_addr;
wire core_busy, core_done;
wire [1:0] core_class;

assign spec_read_addr = core_spec_addr[SPEC_ADDR_BITS-1:0];

cnn_core #(
    .WORD_BITS(WORD_BITS),
    .ADDR_BITS(CORE_ADDR_BITS),
    .KERNEL_FILE(KERNEL_FILE),
    .CBIAS_FILE(CBIAS_FILE),
    .DBIAS_FILE(DBIAS_FILE),
    .DW0_FILE(DW0_FILE),
    .DW1_FILE(DW1_FILE),
    .DW2_FILE(DW2_FILE),
    .DW3_FILE(DW3_FILE)
) u_core (
    .clk(clk),
    .rst_n(rst_n),
    .start(core_start),
    .busy(core_busy),
    .done(core_done),
    .spec_addr(core_spec_addr),
    .spec_data_ch0(spec0_mag),
    .spec_data_ch1(spec1_mag),
    .class_o(core_class)
);

/* Class 1 and 2 swap between this network's ordering and the LED decode
 * the wrapper applies. */
wire [1:0] class_mapped = (core_class == 2'd1) ? 2'd2 :
                          (core_class == 2'd2) ? 2'd1 :
                          core_class;

/* Each spectrogram's done pulse is held until the next arming, because
 * the two channels never finish on the same cycle. */
wire spec0_latched, spec1_latched;

done_latch u_done0 (
    .clk(clk), .rst_n(rst_n),
    .set(spec0_done), .clear(clear_done), .latched(spec0_latched)
);

done_latch u_done1 (
    .clk(clk), .rst_n(rst_n),
    .set(spec1_done), .clear(clear_done), .latched(spec1_latched)
);

cnn_seq_ctrl u_seq (
    .clk(clk),
    .rst_n(rst_n),
    .specs_done(spec0_latched && spec1_latched),
    .core_done(core_done),
    .spec_start(spec_start),
    .clear_done(clear_done),
    .core_start(core_start),
    .emit(emit)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_o <= 1'b0;
        class_o <= 2'd0;
    end else begin
        valid_o <= emit;
        if (emit)
            class_o <= class_mapped;
    end
end

endmodule
