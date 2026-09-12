/*
 * Module: weight_rom
 * Generic synchronous read-only memory for CNN weights/bias, Q8.8.
 * Address in, registered data out one cycle later. Contents load via
 * $readmemh from MEM_FILE when non-empty (real trained weights, hex
 * one value per line); otherwise left for a testbench to preload
 * directly. One instance per weight/bias array (conv weights, conv
 * bias, dense weights, dense bias) -- DEPTH/ADDR_BITS vary per use,
 * the module itself is shared.
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
