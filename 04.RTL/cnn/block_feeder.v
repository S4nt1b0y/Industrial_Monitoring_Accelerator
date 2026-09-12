/*
 * Module: block_feeder
 * Turns the parallel sample block the UART frame buffer produces into
 * the one-sample-per-cycle stream the decimator chain consumes, and
 * carries the block's channel tag alongside each sample.
 *
 * It does not filter by channel: every sample goes out tagged, and the
 * caller gates each per-channel chain on `channel_o`. Walking a block
 * the CNN ignores costs 64 cycles against the ~555,000 the UART takes
 * to deliver the next one, so there is nothing to gain from skipping.
 *
 * Wires block_feeder_ctrl (FSM) to block_feeder_datapath (block
 * register, index counter) -- no logic of its own beyond that.
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
