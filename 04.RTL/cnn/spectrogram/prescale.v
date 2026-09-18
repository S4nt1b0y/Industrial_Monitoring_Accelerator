/*
 * Module: prescale
 * Saturating left shift by SHIFT on the decimated sample stream,
 * between decimator_x64 and spectrogram.
 *
 * After decimation the signal peaks near 81 of 32767, and the FFT
 * scales its output by 1/64. Without the shift the spectrogram keeps
 * about 2 bits of signal. SHIFT = 7 leaves about 3x headroom over the
 * largest measured peak; larger inputs saturate.
 * Combinational: valid and sample pass through in the same cycle.
 * Status: implemented.
 */
module prescale #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter SHIFT     = 7
) (
    input  wire signed [WORD_BITS-1:0] sample_in,
    output wire signed [WORD_BITS-1:0] sample_out
);

localparam signed [WORD_BITS-1:0] SAT_MAX = {1'b0, {(WORD_BITS-1){1'b1}}};
localparam signed [WORD_BITS-1:0] SAT_MIN = {1'b1, {(WORD_BITS-1){1'b0}}};

/* Largest input whose shifted value still fits. */
wire signed [WORD_BITS-1:0] limit_hi = SAT_MAX >>> SHIFT;
wire signed [WORD_BITS-1:0] limit_lo = SAT_MIN >>> SHIFT;

assign sample_out = (sample_in > limit_hi) ? SAT_MAX :
                    (sample_in < limit_lo) ? SAT_MIN :
                    (sample_in <<< SHIFT);

endmodule
