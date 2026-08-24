/*
 * Module: det_q88
 * Determinant of a KxK Q8.8 matrix via recursive row-0 cofactor
 * expansion: for K>=2, instantiates K copies of itself on the
 * (K-1)x(K-1) minors (row 0 removed, column j removed); K==1 is the
 * trivial base case (the single element). Fully combinational.
 * Internal terms/sums use WIDE_BITS (see common/q88_ops.vh) so a
 * well-conditioned matrix's minors don't overflow plain Q8.8. Every multiply/add is
 * still rounded and saturated individually, matching
 * 03.Reference/matrix_inv/algorithms/cofactors.py:det_fixed bit-exactly.
 * Minor extraction is static wire routing: row/column to remove are
 * elaboration-time constants (K and j), so there is no runtime muxing.
 */
module det_q88 #(
    parameter K         = 4,
    parameter WORD_BITS = 16,
    parameter FRAC_BITS = 8,
    parameter WIDE_BITS = 32
) (
    input  wire signed [K*K*WORD_BITS-1:0] a_in,
    output wire signed [WIDE_BITS-1:0]     det_out
);

`include "q88_ops.vh"

generate
if (K == 1) begin : base_case

    assign det_out = q88_sign_extend_to_wide(a_in[WORD_BITS-1:0]);

end else begin : recursive_case

    genvar j, r, c;
    wire signed [WIDE_BITS-1:0] term [0:K-1];
    wire signed [(K-1)*(K-1)*WORD_BITS-1:0] minor [0:K-1];
    wire signed [WIDE_BITS-1:0] partial [0:K-1];

    for (j = 0; j < K; j = j + 1) begin : minors
        for (r = 0; r < K-1; r = r + 1) begin : rows
            for (c = 0; c < K-1; c = c + 1) begin : cols
                // Remove row 0 (source row = r+1) and column j
                // (source col = c, or c+1 past the removed column).
                assign minor[j][(r*(K-1)+c+1)*WORD_BITS-1 -: WORD_BITS] =
                    a_in[((r+1)*K + ((c < j) ? c : c + 1) + 1)*WORD_BITS-1 -: WORD_BITS];
            end
        end

        wire signed [WIDE_BITS-1:0] subdet;
        det_q88 #(
            .K(K-1), .WORD_BITS(WORD_BITS), .FRAC_BITS(FRAC_BITS), .WIDE_BITS(WIDE_BITS)
        ) u_minor (
            .a_in(minor[j]),
            .det_out(subdet)
        );

        wire signed [WIDE_BITS-1:0] signed_subdet = (j % 2 == 0) ? subdet : q88_neg_sat_wide(subdet);
        assign term[j] = q88_mul_wide(a_in[(j+1)*WORD_BITS-1 -: WORD_BITS], signed_subdet);
    end

    assign partial[0] = term[0];
    for (j = 1; j < K; j = j + 1) begin : accum
        assign partial[j] = q88_add_sat_wide(partial[j-1], term[j]);
    end

    assign det_out = partial[K-1];

end
endgenerate

endmodule
