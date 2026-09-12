/*
 * Module: spectrogram
 * One channel of the low-frequency spectrogram the CNN consumes:
 * N_COLS overlapping columns of N_BINS magnitude bins, built from a
 * sliding window of decimated samples. Wires spectrogram_ctrl (FSM) to
 * spectrogram_datapath (window, FFT, magnitude, column memory); no
 * logic of its own beyond that connection.
 *
 * Feed it the output of decimator_x64. For the adopted 2-channel CNN
 * configuration (mancal A: x_A and y_A) instantiate two of these, one
 * per channel.
 * Status: implemented.
 */
module spectrogram #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter N_FFT     = 64,
    parameter N_BINS    = 32,
    parameter N_COLS    = 32,
    parameter HOP       = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        start,
    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,

    output wire                        busy,
    /* High only while a column is being computed. The caller must hold
     * off new samples then: the sliding window is read into the FFT
     * over several cycles and shifting it mid-read corrupts the column.
     * `busy` cannot be used for that -- it is high across the whole
     * frame, including while waiting for the very samples that advance
     * it. */
    output wire                        col_busy,
    output wire                        done,

    input  wire [9:0]                  read_addr,  // col*N_BINS + (bin-1)
    output wire [WORD_BITS-1:0]        read_mag
);

wire run_init, fft_load_step, fft_go, mag_go, mag_store, col_next;
wire col_trigger, load_last, fft_done, mag_done, bin_last, col_last;

spectrogram_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .col_trigger(col_trigger),
    .load_last(load_last),
    .fft_done(fft_done),
    .mag_done(mag_done),
    .bin_last(bin_last),
    .col_last(col_last),
    .busy(busy),
    .col_busy(col_busy),
    .done(done),
    .run_init(run_init),
    .fft_load_step(fft_load_step),
    .fft_go(fft_go),
    .mag_go(mag_go),
    .mag_store(mag_store),
    .col_next(col_next)
);

spectrogram_datapath #(
    .WORD_BITS(WORD_BITS),
    .N_FFT(N_FFT),
    .N_BINS(N_BINS),
    .N_COLS(N_COLS),
    .HOP(HOP)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid),
    .sample_in(sample_in),
    .run_init(run_init),
    .fft_load_step(fft_load_step),
    .fft_go(fft_go),
    .mag_go(mag_go),
    .mag_store(mag_store),
    .col_next(col_next),
    .col_trigger(col_trigger),
    .load_last(load_last),
    .fft_done(fft_done),
    .mag_done(mag_done),
    .bin_last(bin_last),
    .col_last(col_last),
    .read_addr(read_addr),
    .read_mag(read_mag)
);

endmodule
