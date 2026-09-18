/*
 * Module: bundle_rom
 * Read-only memory that outputs a group of GROUP_WORDS consecutive
 * words as one bus, selected by a group index.
 *
 * Used where all words of a group are needed at once: the kernel taps
 * of the current conv2d filter and the dense biases.
 * Word 0 of a group is in the low bits. Combinational read.
 * Contents load with $readmemh from MEM_FILE (hex, one value per line)
 * when MEM_FILE is not empty.
 * Status: implemented.
 */
module bundle_rom #(
    parameter WORD_BITS   = 16,
    parameter GROUP_WORDS = 18,
    parameter N_GROUPS    = 8,
    parameter IDX_BITS    = 3,
    parameter MEM_FILE    = ""
) (
    input  wire [IDX_BITS-1:0] group_idx,
    output wire signed [GROUP_WORDS*WORD_BITS-1:0] group_out
);

localparam DEPTH = N_GROUPS * GROUP_WORDS;

reg signed [WORD_BITS-1:0] mem [0:DEPTH-1];

initial begin
    if (MEM_FILE != "")
        $readmemh(MEM_FILE, mem);
end

genvar i;
generate
    for (i = 0; i < GROUP_WORDS; i = i + 1) begin : word
        assign group_out[(i+1)*WORD_BITS-1 -: WORD_BITS] =
            mem[group_idx * GROUP_WORDS + i];
    end
endgenerate

endmodule
