/*
 * Module: decimator_x64
 * The /64 decimation the CNN spectrogram path needs, built as two
 * cascaded /8 fir_decim stages. Splitting it in two is what makes it
 * implementable: a single /64 stage would need a very long filter to
 * get the same stopband, while two /8 stages of 32 taps each cost
 * ~4.5 MACs per input sample in total.
 *
 * Validated against the trained CNN before any of this RTL was
 * written: 03.Reference/tools/study_hw_decimator.py rebuilt
 * spectrograms with exactly this filter and the classifier's
 * predictions matched the reference decimator on every window tested
 * (02.Architecture/open_decisions.md).
 *
 * The second stage only ever sees one sample per 8 of the first, so
 * the two never compute at the same time and the MAC lanes could be
 * shared between them later if DSP count becomes tight.
 * Status: implemented.
 */
module decimator_x64 #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,  // Q2.30
    parameter FRAC_BITS = 15,
    parameter TAPS      = 32,
    parameter NUM_MACS  = 4
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,

    output wire                        busy,
    output wire signed [WORD_BITS-1:0] sample_out,
    output wire                        sample_out_valid
);

wire                        mid_valid;
wire signed [WORD_BITS-1:0] mid_sample;
wire                        busy1, busy2;

assign busy = busy1 || busy2;

fir_decim #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .TAPS(TAPS),
    .NUM_MACS(NUM_MACS),
    .DECIM(8)
) u_stage1 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid),
    .sample_in(sample_in),
    .busy(busy1),
    .sample_out(mid_sample),
    .sample_out_valid(mid_valid)
);

fir_decim #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .TAPS(TAPS),
    .NUM_MACS(NUM_MACS),
    .DECIM(8)
) u_stage2 (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(mid_valid),
    .sample_in(mid_sample),
    .busy(busy2),
    .sample_out(sample_out),
    .sample_out_valid(sample_out_valid)
);

endmodule
