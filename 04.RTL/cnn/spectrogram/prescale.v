/*
 * Module: prescale
 * Saturating left shift on the decimated sample stream, between
 * decimator_x64 and spectrogram.
 *
 * Decimating by 64 discards the high-frequency energy that dominates
 * these signals, so what reaches the FFT peaks around 81 of 32767 --
 * roughly 5 bits of a 16-bit word. The FFT then halves both butterfly
 * operands at each of its 6 stages, costing another factor of 64, and
 * the spectrogram comes out with about 2 bits of signal left. Measured
 * in 03.Reference/tools/study_hw_quantization.py, that collapses the
 * classifier onto a single class; with this shift it agrees with the
 * float path on 70 of 72 windows.
 *
 * SHIFT=7 was chosen from the measured peak: shifts of 6, 7 and 8 all
 * give the same accuracy, and 7 leaves a factor of ~3 before the
 * loudest recording in the set would clip. The saturation is there for
 * anything louder than what was measured -- clipping degrades, wrapping
 * would invert the sample.
 *
 * Combinational, so the valid/sample handshake passes straight through.
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
