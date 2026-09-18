/*
 * Module: fft
 * 64-point radix-2 DIF FFT of a real input window, Q1.15 complex.
 * Connects fft_ctrl to fft_datapath.
 *
 * Usage: pulse `start`, then stream 64 samples on load_data/load_valid
 * (imaginary part is zero). `done` pulses when the 6 stages finish.
 * Results are read on read_addr in natural bin order, one cycle after
 * the address is set.
 * The output is scaled by 1/64 (see fft_butterfly). The z-score in
 * normalize removes this scale.
 * Status: implemented.
 */
module fft #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,
    parameter FRAC_BITS = 15,
    parameter N         = 64,
    parameter STAGES    = 6
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        start,
    input  wire                        load_valid,
    input  wire signed [WORD_BITS-1:0] load_data,

    output wire                        busy,
    output wire                        done,

    input  wire [5:0]                  read_addr,
    output wire signed [WORD_BITS-1:0] read_re,
    output wire signed [WORD_BITS-1:0] read_im
);

wire run_init, load_en, bf_fetch_a, bf_fetch_b, bf_issue, bf_write_a;
wire bf_store, stage_next;
wire load_done, bf_done, bf_last, stage_last;

fft_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .load_valid(load_valid),
    .load_done(load_done),
    .bf_done(bf_done),
    .bf_last(bf_last),
    .stage_last(stage_last),
    .busy(busy),
    .done(done),
    .run_init(run_init),
    .load_en(load_en),
    .bf_fetch_a(bf_fetch_a),
    .bf_fetch_b(bf_fetch_b),
    .bf_issue(bf_issue),
    .bf_write_a(bf_write_a),
    .bf_store(bf_store),
    .stage_next(stage_next)
);

fft_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .N(N),
    .STAGES(STAGES)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .run_init(run_init),
    .load_en(load_en),
    .bf_fetch_a(bf_fetch_a),
    .bf_fetch_b(bf_fetch_b),
    .bf_issue(bf_issue),
    .bf_write_a(bf_write_a),
    .bf_store(bf_store),
    .stage_next(stage_next),
    .load_data(load_data),
    .load_done(load_done),
    .bf_done(bf_done),
    .bf_last(bf_last),
    .stage_last(stage_last),
    .read_addr(read_addr),
    .read_re(read_re),
    .read_im(read_im)
);

endmodule
