/*
 * Module: block_feeder_datapath
 * Holds one accepted sample block and walks it out one sample per
 * cycle. Sample 0 sits in the low bits, which is how the upstream
 * frame buffer packs them.
 *
 * The block has to be latched, not read straight through: the upstream
 * buffer retires a frame on the same edge it sees ready and valid
 * together, and its output then moves to the next frame.
 * Every state decision comes from block_feeder_ctrl as a one-cycle
 * control pulse; this module never decides what to do next on its own.
 * Status: implemented.
 */
module block_feeder_datapath #(
    parameter WORD_BITS = 16,
    parameter N         = 64,
    parameter IDX_BITS  = 6
) (
    input  wire clk,
    input  wire rst_n,

    input  wire do_load,
    input  wire do_shift,

    output wire idx_last,

    input  wire signed [N*WORD_BITS-1:0] sample_block_i,
    input  wire [1:0]                    channel_i,

    output wire signed [WORD_BITS-1:0]   sample_o,
    output wire [1:0]                    channel_o
);

reg signed [N*WORD_BITS-1:0] block_r;
reg [1:0]                    channel_r;
reg [IDX_BITS-1:0]           idx;

assign idx_last  = (idx == N-1);
assign sample_o  = block_r[idx*WORD_BITS +: WORD_BITS];
assign channel_o = channel_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        block_r   <= {N*WORD_BITS{1'b0}};
        channel_r <= 2'd0;
        idx       <= {IDX_BITS{1'b0}};
    end else begin
        if (do_load) begin
            block_r   <= sample_block_i;
            channel_r <= channel_i;
            idx       <= {IDX_BITS{1'b0}};
        end else if (do_shift) begin
            idx <= idx + {{(IDX_BITS-1){1'b0}}, 1'b1};
        end
    end
end

endmodule
