/*
 * Module: fft_ctrl
 * FSM for fft_datapath: load 64 samples, then run 6 stages of 32
 * butterflies, one butterfly at a time.
 *
 * Each butterfly: fetch a, fetch b, compute, write a, write b. The
 * banks have one read and one write port, with one cycle of read
 * latency.
 * stage_next is not pulsed after the last stage, so the bank selector
 * still points to the bank holding the result.
 * One transform takes 1,800 cycles.
 * Status: implemented.
 */
module fft_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire load_valid,
    input  wire load_done,
    input  wire bf_done,
    input  wire bf_last,
    input  wire stage_last,

    output reg  busy,
    output reg  done,
    output reg  run_init,
    output reg  load_en,
    output reg  bf_fetch_a,
    output reg  bf_fetch_b,
    output reg  bf_issue,
    output reg  bf_write_a,
    output reg  bf_store,
    output reg  stage_next
);

localparam S_IDLE       = 4'd0,
           S_LOAD       = 4'd1,
           S_BF_FETCH_A = 4'd2,
           S_BF_FETCH_B = 4'd3,
           S_BF_ISSUE   = 4'd4,
           S_BF_WAIT    = 4'd5,
           S_BF_WRITE_A = 4'd6,
           S_BF_STORE   = 4'd7,
           S_STAGE      = 4'd8,
           S_DONE       = 4'd9;

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
        S_IDLE:       next_state = start ? S_LOAD : S_IDLE;
        S_LOAD:       next_state = load_done ? S_BF_FETCH_A : S_LOAD;
        S_BF_FETCH_A: next_state = S_BF_FETCH_B;
        S_BF_FETCH_B: next_state = S_BF_ISSUE;
        S_BF_ISSUE:   next_state = S_BF_WAIT;
        S_BF_WAIT:    next_state = bf_done ? S_BF_WRITE_A : S_BF_WAIT;
        S_BF_WRITE_A: next_state = S_BF_STORE;
        S_BF_STORE:   next_state = bf_last ? S_STAGE : S_BF_FETCH_A;
        S_STAGE:      next_state = stage_last ? S_DONE : S_BF_FETCH_A;
        S_DONE:       next_state = S_IDLE;
        default:      next_state = S_IDLE;
    endcase
end

/* output logic */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    run_init   = (state == S_IDLE) && start;
    load_en    = (state == S_LOAD) && load_valid;
    bf_fetch_a = (state == S_BF_FETCH_A);
    bf_fetch_b = (state == S_BF_FETCH_B);
    bf_issue   = (state == S_BF_ISSUE);
    bf_write_a = (state == S_BF_WRITE_A);
    bf_store   = (state == S_BF_STORE);
    stage_next = (state == S_STAGE) && !stage_last;
end

endmodule
