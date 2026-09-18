/*
 * Module: block_feeder_datapath
 * Holds one accepted sample block and outputs it one sample per cycle.
 *
 * Sample 0 is in the low bits, as packed by uart_frame_buffer.
 * The block is registered because uart_frame_buffer moves to the next
 * frame on the cycle the block is accepted.
 * Control pulses come from block_feeder_ctrl.
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
