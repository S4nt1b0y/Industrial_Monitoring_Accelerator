/*
 * Module: ram
 * Memory with one synchronous write port and one registered read port.
 *
 * The registered read lets the array map to M10K. A combinational read
 * is built from registers instead.
 * Read latency: one cycle.
 * Status: implemented.
 */
module ram #(
    parameter WORD_BITS = 16,
    parameter DEPTH     = 1024,
    parameter ADDR_BITS = 12
) (
    input  wire clk,

    input  wire [ADDR_BITS-1:0]        wr_addr,
    input  wire signed [WORD_BITS-1:0] wr_data,
    input  wire                        wr_en,

    input  wire [ADDR_BITS-1:0]        rd_addr,
    output wire signed [WORD_BITS-1:0] rd_data
);

reg signed [WORD_BITS-1:0] mem [0:DEPTH-1];
reg signed [WORD_BITS-1:0] rd_data_r;

assign rd_data = rd_data_r;

/* Read-before-write on an address collision. Callers never read and
 * write the same address in the same cycle. */
always @(posedge clk) begin
    if (wr_en)
        mem[wr_addr] <= wr_data;
    rd_data_r <= mem[rd_addr];
end

endmodule
