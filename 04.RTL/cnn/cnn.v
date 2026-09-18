/*
 * Module: cnn
 * Block interface for cnn_path, with the port list top_wrapper uses:
 * 64-sample blocks in, class out, valid/ready handshake.
 *
 * block_feeder converts each block into the sample stream cnn_path
 * takes.
 * Channel tags follow the transmitter order x_A, y_A, x_B, y_B. Tags 0
 * and 1 are processed. Tags 2 and 3 are accepted and discarded, with
 * ready_o kept high so ingestion does not stall.
 * ml_pipeline_fsm names tag 1 CH_X_B; this module follows the
 * transmitter order.
 * Status: implemented.
 */
module cnn #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    /* Passed to cnn_path; see its header. */
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

    input  wire signed [N*DATA_WIDTH-1:0] sample_block_i,
    input  wire [1:0]                     channel_i,
    input  wire                           valid_i,
    output wire                           ready_o,

    output wire                           valid_o,
    output wire [1:0]                     class_o
);

localparam WORD_BITS = DATA_WIDTH;

wire                        sample_valid;
wire signed [WORD_BITS-1:0] sample;
wire [1:0]                  sample_channel;
wire                        stall;

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
    .dst_busy(stall),
    .sample_valid_o(sample_valid),
    .sample_o(sample),
    .channel_o(sample_channel)
);

cnn_path #(
    .WORD_BITS(WORD_BITS),
    .KERNEL_FILE(KERNEL_FILE),
    .CBIAS_FILE(CBIAS_FILE),
    .DBIAS_FILE(DBIAS_FILE),
    .DW0_FILE(DW0_FILE),
    .DW1_FILE(DW1_FILE),
    .DW2_FILE(DW2_FILE),
    .DW3_FILE(DW3_FILE)
) u_path (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid),
    .sample_in(sample),
    .sample_channel(sample_channel),
    .stall(stall),
    .valid_o(valid_o),
    .class_o(class_o)
);

endmodule
