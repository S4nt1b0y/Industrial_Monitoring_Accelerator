/*
 * Module: spectrogram_ctrl
 * FSM for spectrogram_datapath: wait for the hop counter to request a
 * column, load the window into the FFT, then pass bins 1..N_BINS
 * through the magnitude unit and store each one. The frame is complete
 * after N_COLS columns.
 *
 * The FFT result port has one cycle of read latency, so each bin has a
 * fetch cycle (S_MAG_FETCH) before the magnitude starts.
 * Status: implemented.
 */
module spectrogram_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire col_trigger,
    input  wire load_last,
    input  wire fft_done,
    input  wire mag_done,
    input  wire bin_last,
    input  wire col_last,

    output reg  busy,
    output reg  col_busy,
    output reg  done,
    output reg  run_init,
    output reg  fft_load_step,
    output reg  fft_go,
    output reg  mag_go,
    output reg  mag_store,
    output reg  col_next
);

localparam S_ARMED     = 4'd0,
           S_FFT_START = 4'd1,
           S_FFT_LOAD  = 4'd2,
           S_FFT_WAIT  = 4'd3,
           S_MAG_FETCH = 4'd4,
           S_MAG_START = 4'd5,
           S_MAG_WAIT  = 4'd6,
           S_MAG_STORE = 4'd7,
           S_COL_NEXT  = 4'd8,
           S_DONE      = 4'd9,
           S_IDLE      = 4'd10;

reg [3:0] state, next_state;

/* state register */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= next_state;
end

/* next-state logic */
always @* begin
    case (state)
        S_IDLE:      next_state = start ? S_ARMED : S_IDLE;
        S_ARMED:     next_state = col_trigger ? S_FFT_START : S_ARMED;
        S_FFT_START: next_state = S_FFT_LOAD;
        S_FFT_LOAD:  next_state = load_last ? S_FFT_WAIT : S_FFT_LOAD;
        S_FFT_WAIT:  next_state = fft_done ? S_MAG_FETCH : S_FFT_WAIT;
        S_MAG_FETCH: next_state = S_MAG_START;
        S_MAG_START: next_state = S_MAG_WAIT;
        S_MAG_WAIT:  next_state = mag_done ? S_MAG_STORE : S_MAG_WAIT;
        S_MAG_STORE: next_state = bin_last ? S_COL_NEXT : S_MAG_FETCH;
        S_COL_NEXT:  next_state = col_last ? S_DONE : S_ARMED;
        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

/* output logic */
always @* begin
    busy          = (state != S_IDLE);
    /* Column in progress: the window is being loaded into the FFT.
     * `busy` also covers S_ARMED, where new samples are expected. */
    col_busy      = (state != S_IDLE) && (state != S_ARMED);
    done          = (state == S_DONE);
    run_init      = (state == S_IDLE) && start;
    fft_go        = (state == S_FFT_START);
    fft_load_step = (state == S_FFT_LOAD);
    mag_go        = (state == S_MAG_START);
    mag_store     = (state == S_MAG_STORE);
    col_next      = (state == S_COL_NEXT);
end

endmodule
