/*
 * Module: cnn_core
 * The classifier itself: normalized spectrogram in, class index out.
 * Wires cnn_core_ctrl (FSM) and cnn_core_datapath (counters, flatten
 * offset) to the compute blocks and the memories between them.
 *
 * Dataflow, one pass per classification:
 *   spectrogram memories (Q1.15, one per channel, read here)
 *     -> normalize      one pass per channel -> act memories (Q8.8)
 *     -> conv2d + ReLU  once per filter      -> conv memory
 *     -> maxpool 2x2    once per filter      -> flat memory at
 *                                              filter*POOL_ELEMS
 *     -> dense          one pass over all filters
 *     -> argmax         combinational
 *
 * The convolution unit is instantiated once and reused across the 8
 * filters; see cnn_core_ctrl's header for why serialising is the right
 * trade here. The spectrogram memories stay outside this module because
 * they belong to the spectrogram blocks that fill them.
 * Status: implemented.
 */
module cnn_core #(
    parameter WORD_BITS   = 16,
    parameter ACC_BITS    = 32,
    parameter FRAC_BITS   = 8,
    parameter IMG_H       = 32,
    parameter IMG_W       = 32,
    parameter KSIZE       = 3,
    parameter N_CHANNELS  = 2,
    parameter N_FILTERS   = 8,
    parameter POOL        = 2,
    parameter N_CLASSES   = 4,
    parameter ADDR_BITS   = 12,
    parameter GAIN        = 287,
    parameter OFFSET      = 116,
    parameter KERNEL_FILE = "",
    parameter CBIAS_FILE  = "",
    parameter DBIAS_FILE  = "",
    parameter DW0_FILE    = "",
    parameter DW1_FILE    = "",
    parameter DW2_FILE    = "",
    parameter DW3_FILE    = ""
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output wire busy,
    output wire done,

    output wire [ADDR_BITS-1:0] spec_addr,
    input  wire [WORD_BITS-1:0] spec_data_ch0,
    input  wire [WORD_BITS-1:0] spec_data_ch1,

    output reg  [1:0] class_o
);

localparam IMG_ELEMS  = IMG_H * IMG_W;             // 1024
localparam POOL_H     = IMG_H / POOL;              // 16
localparam POOL_W     = IMG_W / POOL;              // 16
localparam POOL_ELEMS = POOL_H * POOL_W;           // 256
localparam FLAT_ELEMS = N_FILTERS * POOL_ELEMS;    // 2048
localparam K_WORDS    = KSIZE * KSIZE * N_CHANNELS; // 18
localparam FILT_BITS  = 3;
localparam CHAN_BITS  = 1;

wire norm_start, conv_start, pool_start, dense_start, latch_class;
wire chan_init, chan_step, filter_init, filter_step;
wire norm_done, conv_done, pool_done, dense_done;
wire chan_last, filter_last;
wire [CHAN_BITS-1:0] chan_idx;
wire [FILT_BITS-1:0] filter_idx;

cnn_core_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .norm_done(norm_done),
    .conv_done(conv_done),
    .pool_done(pool_done),
    .dense_done(dense_done),
    .chan_last(chan_last),
    .filter_last(filter_last),
    .busy(busy),
    .done(done),
    .chan_init(chan_init),
    .chan_step(chan_step),
    .filter_init(filter_init),
    .filter_step(filter_step),
    .norm_start(norm_start),
    .conv_start(conv_start),
    .pool_start(pool_start),
    .dense_start(dense_start),
    .latch_class(latch_class)
);

wire [ADDR_BITS-1:0] pool_out_addr;
wire [ADDR_BITS-1:0] flat_addr;

cnn_core_datapath #(
    .N_FILTERS(N_FILTERS),
    .N_CHANNELS(N_CHANNELS),
    .POOL_ELEMS(POOL_ELEMS),
    .FILT_BITS(FILT_BITS),
    .CHAN_BITS(CHAN_BITS),
    .ADDR_BITS(ADDR_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .chan_init(chan_init),
    .chan_step(chan_step),
    .filter_init(filter_init),
    .filter_step(filter_step),
    .chan_idx(chan_idx),
    .chan_last(chan_last),
    .filter_idx(filter_idx),
    .filter_last(filter_last),
    .pool_out_addr(pool_out_addr),
    .flat_addr(flat_addr)
);

/* ---- normalize: one pass per channel, Q1.15 spectrogram -> Q8.8 ---- */
wire [ADDR_BITS-1:0]        norm_in_addr, norm_out_addr;
wire signed [WORD_BITS-1:0] norm_out_word;
wire                        norm_out_we;

assign spec_addr = norm_in_addr;
wire [WORD_BITS-1:0] norm_in_data = chan_idx[0] ? spec_data_ch1 : spec_data_ch0;

normalize #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .N_ELEMS(IMG_ELEMS),
    .ADDR_BITS(ADDR_BITS),
    .GAIN(GAIN),
    .OFFSET(OFFSET)
) u_normalize (
    .clk(clk),
    .rst_n(rst_n),
    .start(norm_start),
    .busy(),
    .done(norm_done),
    .in_addr(norm_in_addr),
    .in_data(norm_in_data),
    .out_addr(norm_out_addr),
    .out_word(norm_out_word),
    .out_we(norm_out_we)
);

/* ---- activation memories: one per channel, read together by conv ---- */
wire [ADDR_BITS-1:0] img_addr;
wire signed [WORD_BITS-1:0] act_ch0, act_ch1;
wire signed [N_CHANNELS*WORD_BITS-1:0] img_data = {act_ch1, act_ch0};

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_ELEMS), .ADDR_BITS(ADDR_BITS)) u_act_ch0 (
    .clk(clk),
    .wr_addr(norm_out_addr),
    .wr_data(norm_out_word),
    .wr_en(norm_out_we && (chan_idx == 1'b0)),
    .rd_addr(img_addr),
    .rd_data(act_ch0)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_ELEMS), .ADDR_BITS(ADDR_BITS)) u_act_ch1 (
    .clk(clk),
    .wr_addr(norm_out_addr),
    .wr_data(norm_out_word),
    .wr_en(norm_out_we && (chan_idx == 1'b1)),
    .rd_addr(img_addr),
    .rd_data(act_ch1)
);

/* ---- convolution: one unit, reused once per filter ---- */
wire signed [K_WORDS*WORD_BITS-1:0] kernel_in;
wire signed [WORD_BITS-1:0] conv_bias;
wire [ADDR_BITS-1:0] conv_out_addr;
wire signed [WORD_BITS-1:0] conv_out_word;
wire conv_out_we;

bundle_rom #(
    .WORD_BITS(WORD_BITS),
    .GROUP_WORDS(K_WORDS),
    .N_GROUPS(N_FILTERS),
    .IDX_BITS(FILT_BITS),
    .MEM_FILE(KERNEL_FILE)
) u_kernel_rom (
    .group_idx(filter_idx),
    .group_out(kernel_in)
);

weight_rom #(
    .WORD_BITS(WORD_BITS),
    .DEPTH(N_FILTERS),
    .ADDR_BITS(FILT_BITS),
    .MEM_FILE(CBIAS_FILE)
) u_conv_bias_rom (
    .clk(clk),
    .addr(filter_idx),
    .data(conv_bias)
);

conv2d_top #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .KSIZE(KSIZE),
    .CIN(N_CHANNELS),
    .IMG_H(IMG_H),
    .IMG_W(IMG_W),
    .ADDR_BITS(ADDR_BITS)
) u_conv2d (
    .clk(clk),
    .rst_n(rst_n),
    .start(conv_start),
    .busy(),
    .done(conv_done),
    .img_addr(img_addr),
    .img_data(img_data),
    .kernel_in(kernel_in),
    .bias_in(conv_bias),
    .out_addr(conv_out_addr),
    .out_word(conv_out_word),
    .out_we(conv_out_we)
);

/* ---- feature map of the filter being processed ---- */
wire [ADDR_BITS-1:0] pool_in_addr;
wire signed [WORD_BITS-1:0] pool_in_data;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_ELEMS), .ADDR_BITS(ADDR_BITS)) u_conv_ram (
    .clk(clk),
    .wr_addr(conv_out_addr),
    .wr_data(conv_out_word),
    .wr_en(conv_out_we),
    .rd_addr(pool_in_addr),
    .rd_data(pool_in_data)
);

/* ---- pooling, written straight into the flattened activation map ---- */
wire signed [WORD_BITS-1:0] pool_out_word;
wire pool_out_we;

maxpool #(
    .WORD_BITS(WORD_BITS),
    .POOL(POOL),
    .IMG_H(IMG_H),
    .IMG_W(IMG_W),
    .ADDR_BITS(ADDR_BITS)
) u_maxpool (
    .clk(clk),
    .rst_n(rst_n),
    .start(pool_start),
    .busy(),
    .done(pool_done),
    .in_addr(pool_in_addr),
    .in_data(pool_in_data),
    .out_addr(pool_out_addr),
    .out_word(pool_out_word),
    .out_we(pool_out_we)
);

wire [ADDR_BITS-1:0] dense_in_addr;
wire signed [WORD_BITS-1:0] dense_in_data;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(FLAT_ELEMS), .ADDR_BITS(ADDR_BITS)) u_flat_ram (
    .clk(clk),
    .wr_addr(flat_addr),
    .wr_data(pool_out_word),
    .wr_en(pool_out_we),
    .rd_addr(dense_in_addr),
    .rd_data(dense_in_data)
);

/* ---- dense layer: one weight memory per class, read in parallel ---- */
wire [ADDR_BITS-1:0] dense_w_addr;
wire signed [WORD_BITS-1:0] dw0, dw1, dw2, dw3;
wire signed [N_CLASSES*WORD_BITS-1:0] dense_w_data = {dw3, dw2, dw1, dw0};
wire signed [N_CLASSES*WORD_BITS-1:0] dense_bias;
wire signed [N_CLASSES*WORD_BITS-1:0] logits;

weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(FLAT_ELEMS), .ADDR_BITS(ADDR_BITS),
             .MEM_FILE(DW0_FILE))
    u_dw0 (.clk(clk), .addr(dense_w_addr), .data(dw0));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(FLAT_ELEMS), .ADDR_BITS(ADDR_BITS),
             .MEM_FILE(DW1_FILE))
    u_dw1 (.clk(clk), .addr(dense_w_addr), .data(dw1));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(FLAT_ELEMS), .ADDR_BITS(ADDR_BITS),
             .MEM_FILE(DW2_FILE))
    u_dw2 (.clk(clk), .addr(dense_w_addr), .data(dw2));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(FLAT_ELEMS), .ADDR_BITS(ADDR_BITS),
             .MEM_FILE(DW3_FILE))
    u_dw3 (.clk(clk), .addr(dense_w_addr), .data(dw3));

bundle_rom #(
    .WORD_BITS(WORD_BITS),
    .GROUP_WORDS(N_CLASSES),
    .N_GROUPS(1),
    .IDX_BITS(1),
    .MEM_FILE(DBIAS_FILE)
) u_dense_bias_rom (
    .group_idx(1'b0),
    .group_out(dense_bias)
);

dense #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .N_IN(FLAT_ELEMS),
    .N_CLASSES(N_CLASSES),
    .ADDR_BITS(ADDR_BITS)
) u_dense (
    .clk(clk),
    .rst_n(rst_n),
    .start(dense_start),
    .busy(),
    .done(dense_done),
    .in_addr(dense_in_addr),
    .in_data(dense_in_data),
    .w_addr(dense_w_addr),
    .w_data(dense_w_data),
    .bias_in(dense_bias),
    .logits(logits)
);

wire [1:0] class_next;

argmax #(.WORD_BITS(WORD_BITS)) u_argmax (
    .logits(logits),
    .class_idx(class_next)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        class_o <= 2'd0;
    else if (latch_class)
        class_o <= class_next;
end

endmodule
