/*
 * Module: block_feeder
 * Converts a parallel 64-sample block into a stream of one sample per
 * cycle, each tagged with the block's channel.
 *
 * All channels are streamed; the caller selects channels by the tag.
 * Connects block_feeder_ctrl to block_feeder_datapath.
 * Status: implemented.
 */
module block_feeder #(
    parameter WORD_BITS = 16,
    parameter N         = 64,
    parameter IDX_BITS  = 6
) (
    input  wire clk,
    input  wire rst_n,

    input  wire signed [N*WORD_BITS-1:0] sample_block_i,
    input  wire [1:0]                    channel_i,
    input  wire                          valid_i,
    output wire                          ready_o,

    input  wire                          dst_busy,
    output wire                          sample_valid_o,
    output wire signed [WORD_BITS-1:0]   sample_o,
    output wire [1:0]                    channel_o
);

wire do_load, do_shift, idx_last;

assign sample_valid_o = do_shift;

block_feeder_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .valid_i(valid_i),
    .dst_busy(dst_busy),
    .idx_last(idx_last),
    .ready_o(ready_o),
    .do_load(do_load),
    .do_shift(do_shift)
);

block_feeder_datapath #(
    .WORD_BITS(WORD_BITS),
    .N(N),
    .IDX_BITS(IDX_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .do_load(do_load),
    .do_shift(do_shift),
    .idx_last(idx_last),
    .sample_block_i(sample_block_i),
    .channel_i(channel_i),
    .sample_o(sample_o),
    .channel_o(channel_o)
);

endmodule
