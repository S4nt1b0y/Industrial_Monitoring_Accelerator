/*
 * Module: bundle_rom
 * Read-only memory that presents a whole group of GROUP_WORDS
 * consecutive words as one bus, selected by a group index.
 *
 * weight_rom.v hands out one word per cycle, which suits the dense
 * layer. Some consumers instead need every word of a group present at
 * once for the whole run: conv2d holds all KSIZE*KSIZE*CIN taps of the
 * current filter on `kernel_in`, and the dense layer holds all
 * N_CLASSES biases. Both are small enough to come out combinationally.
 *
 * Word 0 of a group sits in the low bits. Contents load via $readmemh
 * from MEM_FILE when non-empty (hex, one value per line, written by
 * 03.Reference/tools/export_cnn_weights_hex.py); otherwise left for a
 * testbench to preload directly.
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
