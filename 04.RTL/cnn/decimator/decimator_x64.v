/*
 * Module: decimator_x64
 * Decimation by 64 for the CNN path: two cascaded fir_decim stages of 8.
 *
 * Two /8 stages of 32 taps cost 4.5 MACs per input sample. A single /64
 * stage would need a much longer filter for the same stopband.
 * The second stage runs once per 8 first-stage outputs, so the two
 * stages never compute in the same cycle.
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
