/*
 * Module: cnn_path
 * CNN classification chain: stream of channel-tagged samples in, class
 * index out. cnn.v adds a block interface in front of it.
 *
 * Chain:
 *   decimator_x64    /64, one instance per channel used
 *   prescale         saturating shift left by 7
 *   spectrogram      32 columns x 32 bins, one per channel
 *   cnn_core         normalize, conv/ReLU/pool x8, dense, argmax
 *
 * Input: one sample per cycle with its channel tag. Channel 0 is x_A
 * and channel 1 is y_A (bearing A). Other tags are ignored.
 *
 * `stall`: hold new samples while high. It is high while a decimator
 * computes a dot product or a spectrogram loads a column into the FFT.
 *
 * Class order: the network outputs normal, unbalance, misalignment,
 * wear; top_wrapper decodes normal, misalignment, unbalance, wear.
 * Classes 1 and 2 are swapped at the output.
 *
 * The first classification needs 576 decimated samples per channel
 * (36,864 raw samples).
 * Status: implemented.
 */
module cnn_path #(
    parameter WORD_BITS   = 16,
    /* $readmemh resolves these paths against the tool's working
     * directory. Override them when the tool runs from another
     * directory. */
    parameter KERNEL_FILE = "./weights/conv1_kernels.hex",
    parameter CBIAS_FILE  = "./weights/conv1_bias.hex",
    parameter DBIAS_FILE  = "./weights/dense_b.hex",
    parameter DW0_FILE    = "./weights/dense_w_c0.hex",
    parameter DW1_FILE    = "./weights/dense_w_c1.hex",
    parameter DW2_FILE    = "./weights/dense_w_c2.hex",
    parameter DW3_FILE    = "./weights/dense_w_c3.hex"
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,
    input  wire [1:0]                  sample_channel,
    output wire                        stall,

    output reg                         valid_o,
    output reg  [1:0]                  class_o
);

localparam SPEC_ADDR_BITS = 10;
localparam CORE_ADDR_BITS = 12;
localparam PRESCALE_SHIFT = 7;

localparam CH_X_A = 2'd0;
localparam CH_Y_A = 2'd1;

wire dec0_busy, dec1_busy;
wire spec0_col_busy, spec1_col_busy;

assign stall = dec0_busy || dec1_busy || spec0_col_busy || spec1_col_busy;

/* channel 0: x_A */
wire dec0_valid;
wire signed [WORD_BITS-1:0] dec0_sample, pre0_sample;

decimator_x64 #(.WORD_BITS(WORD_BITS)) u_dec0 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid && (sample_channel == CH_X_A)),
    .sample_in(sample_in),
    .busy(dec0_busy),
    .sample_out(dec0_sample),
    .sample_out_valid(dec0_valid)
);

prescale #(.WORD_BITS(WORD_BITS), .SHIFT(PRESCALE_SHIFT)) u_pre0 (
    .sample_in(dec0_sample),
    .sample_out(pre0_sample)
);

/* channel 1: y_A */
wire dec1_valid;
wire signed [WORD_BITS-1:0] dec1_sample, pre1_sample;

decimator_x64 #(.WORD_BITS(WORD_BITS)) u_dec1 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid && (sample_channel == CH_Y_A)),
    .sample_in(sample_in),
    .busy(dec1_busy),
    .sample_out(dec1_sample),
    .sample_out_valid(dec1_valid)
);

prescale #(.WORD_BITS(WORD_BITS), .SHIFT(PRESCALE_SHIFT)) u_pre1 (
    .sample_in(dec1_sample),
    .sample_out(pre1_sample)
);

/* spectrograms */
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

/* classifier */
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

/* Swap classes 1 and 2 to the order top_wrapper decodes. */
wire [1:0] class_mapped = (core_class == 2'd1) ? 2'd2 :
                          (core_class == 2'd2) ? 2'd1 :
                          core_class;

/* Hold each spectrogram's done pulse until the next arming; the two
 * channels finish on different cycles. */
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
