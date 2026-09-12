/*
 * Module: argmax
 * Index of the largest of N_CLASSES logits, as a two-level comparison
 * tree. No softmax: the reference model classifies straight off the
 * logits, and a monotonic function would not change which one wins.
 *
 * Ties resolve to the lowest index, matching numpy's argmax in
 * 03.Reference/cnn/reference.py -- each comparison only moves to the
 * later candidate when it is strictly greater.
 * Combinational: the caller registers the result along with its own
 * valid pulse.
 * Status: implemented.
 */
module argmax #(
    parameter WORD_BITS = 16
) (
    input  wire signed [4*WORD_BITS-1:0] logits,
    output wire [1:0]                    class_idx
);

wire signed [WORD_BITS-1:0] l0 = logits[0*WORD_BITS +: WORD_BITS];
wire signed [WORD_BITS-1:0] l1 = logits[1*WORD_BITS +: WORD_BITS];
wire signed [WORD_BITS-1:0] l2 = logits[2*WORD_BITS +: WORD_BITS];
wire signed [WORD_BITS-1:0] l3 = logits[3*WORD_BITS +: WORD_BITS];

wire lo_take_1 = (l1 > l0);
wire hi_take_3 = (l3 > l2);

wire signed [WORD_BITS-1:0] lo_val = lo_take_1 ? l1 : l0;
wire signed [WORD_BITS-1:0] hi_val = hi_take_3 ? l3 : l2;

wire [1:0] lo_idx = lo_take_1 ? 2'd1 : 2'd0;
wire [1:0] hi_idx = hi_take_3 ? 2'd3 : 2'd2;

assign class_idx = (hi_val > lo_val) ? hi_idx : lo_idx;

endmodule
