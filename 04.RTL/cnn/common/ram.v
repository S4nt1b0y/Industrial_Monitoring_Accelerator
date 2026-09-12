/*
 * Module: ram
 * Simple memory with one synchronous write port and one combinational
 * read port, shared by the blocks that pass feature maps around
 * (conv2d writes, maxpool reads, and so on).
 *
 * The read is combinational to match spectrogram.v's existing read
 * port, so every producer/consumer pair in this design behaves the
 * same way. Inferring block RAM instead requires a registered read,
 * which would add a wait state to each consumer's tap loop -- a
 * synthesis-phase change to make across all of them at once, not one
 * module at a time.
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

assign rd_data = mem[rd_addr];

always @(posedge clk) begin
    if (wr_en)
        mem[wr_addr] <= wr_data;
end

endmodule
