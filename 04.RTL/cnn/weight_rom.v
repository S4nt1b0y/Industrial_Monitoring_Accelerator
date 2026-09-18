/*
 * Module: weight_rom
 * Read-only memory for CNN weights and biases, Q8.8, with a registered
 * read (one cycle of latency).
 *
 * Contents load with $readmemh from MEM_FILE (hex, one value per line)
 * when MEM_FILE is not empty. One instance per weight or bias array.
 * Status: implemented.
 */
module weight_rom #(
    parameter WORD_BITS = 16,
    parameter DEPTH     = 144,
    parameter ADDR_BITS = 8,    // must satisfy 2**ADDR_BITS >= DEPTH
    parameter MEM_FILE  = ""
) (
    input  wire                        clk,
    input  wire [ADDR_BITS-1:0]        addr,
    output reg  signed [WORD_BITS-1:0] data
);

reg signed [WORD_BITS-1:0] mem [0:DEPTH-1];

initial begin
    if (MEM_FILE != "")
        $readmemh(MEM_FILE, mem);
end

always @(posedge clk) begin
    data <= mem[addr];
end

endmodule
