/*
 * Module: spectrogram_datapath
 * Sliding window, FFT, magnitude and column memory for one channel of
 * the low-frequency spectrogram.
 *
 * Buffering happens AFTER decimation, which is the whole point: the
 * window is 64 decimated samples (~128 bytes) instead of the 35,840
 * raw samples the same time span would need. A new column is taken
 * every HOP decimated samples, so consecutive columns overlap -- the
 * scheme the reference model uses (LOWFREQ_HOP_DECIMATED = 16).
 *
 * Per column: the 64 buffered samples go into the FFT oldest-first,
 * then bins 1..32 (DC dropped, as in the reference) are converted to
 * magnitude and written to the column memory.
 *
 * Assumes decimated samples arrive far apart relative to a column's
 * compute time (~1400 cycles against tens of thousands at 400 Hz), so
 * the window is never shifted mid-transform. `busy` is exported.
 *
 * Every state decision comes from spectrogram_ctrl as a control pulse.
 * Status: implemented.
 */
module spectrogram_datapath #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter N_FFT     = 64,
    parameter N_BINS    = 32,  // bins 1..32
    parameter N_COLS    = 32,
    parameter HOP       = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,

    input  wire run_init,
    input  wire fft_load_step,
    input  wire fft_go,
    input  wire mag_go,
    input  wire mag_store,
    input  wire col_next,

    output wire col_trigger,
    output wire load_last,
    output wire fft_done,
    output wire mag_done,
    output wire bin_last,
    output wire col_last,

    input  wire [9:0]           read_addr,   // col*N_BINS + (bin-1)
    output wire [WORD_BITS-1:0] read_mag
);

integer i;

reg signed [WORD_BITS-1:0] window [0:N_FFT-1];  // [0] newest, [N_FFT-1] oldest
reg [6:0] primed;      // counts up to N_FFT while the window fills
reg [4:0] hop_count;
reg [6:0] load_index;  // 0..N_FFT-1, feeds the FFT oldest-first
reg [5:0] bin_index;   // 1..N_BINS
reg [5:0] col_index;   // 0..N_COLS-1

reg [WORD_BITS-1:0] spec_mem [0:N_COLS*N_BINS-1];

wire window_full = (primed == N_FFT);
assign col_trigger = sample_valid && window_full && (hop_count == HOP-1);
assign load_last   = (load_index == N_FFT-1);
assign bin_last    = (bin_index == N_BINS);
assign col_last    = (col_index == N_COLS-1);

assign read_mag = spec_mem[read_addr];

/* FFT: fed oldest-first, so index N_FFT-1 down to 0 */
wire signed [WORD_BITS-1:0] fft_load_data = window[N_FFT-1 - load_index];

wire signed [WORD_BITS-1:0] fft_re, fft_im;
wire fft_busy;

fft #(
    .WORD_BITS(WORD_BITS),
    .N(N_FFT)
) u_fft (
    .clk(clk),
    .rst_n(rst_n),
    .start(fft_go),
    .load_valid(fft_load_step),
    .load_data(fft_load_data),
    .busy(fft_busy),
    .done(fft_done),
    .read_addr(bin_index[5:0]),
    .read_re(fft_re),
    .read_im(fft_im)
);

wire [WORD_BITS-1:0] mag_value;
wire mag_busy;

magnitude #(
    .WORD_BITS(WORD_BITS)
) u_magnitude (
    .clk(clk),
    .rst_n(rst_n),
    .start(mag_go),
    .re(fft_re),
    .im(fft_im),
    .busy(mag_busy),
    .done(mag_done),
    .mag(mag_value)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < N_FFT; i = i + 1)
            window[i] <= {WORD_BITS{1'b0}};
        primed     <= 7'd0;
        hop_count  <= 5'd0;
        load_index <= 7'd0;
        bin_index  <= 6'd1;
        col_index  <= 6'd0;
    end else begin
        if (run_init) begin
            col_index  <= 6'd0;
            bin_index  <= 6'd1;
            load_index <= 7'd0;
        end

        if (sample_valid) begin
            for (i = N_FFT-1; i > 0; i = i - 1)
                window[i] <= window[i-1];
            window[0] <= sample_in;

            if (!window_full)
                primed <= primed + 7'd1;

            hop_count <= (hop_count == HOP-1) ? 5'd0 : hop_count + 5'd1;
        end

        if (fft_go)
            load_index <= 7'd0;
        else if (fft_load_step && !load_last)
            load_index <= load_index + 7'd1;

        if (mag_store) begin
            spec_mem[col_index*N_BINS + (bin_index - 1)] <= mag_value;
            bin_index <= bin_last ? 6'd1 : bin_index + 6'd1;
        end

        if (col_next)
            col_index <= col_last ? 6'd0 : col_index + 6'd1;
    end
end

endmodule
