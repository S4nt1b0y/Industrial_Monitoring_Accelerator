/*
 * Module: fft_ctrl
 * FSM for fft_datapath: load 64 samples, then run 6 stages of 32
 * butterflies each, one butterfly at a time. Three separate always
 * blocks: state register, next-state (transition) logic, and output
 * control-pulse logic -- never mixed.
 *
 * stage_next is deliberately NOT pulsed after the final stage, so the
 * bank selector still points at that stage's destination and the read
 * port can find the result.
 *
 * One butterfly at a time is enough by a wide margin: ~768 cycles per
 * transform against the tens of thousands available between spectrogram
 * columns. More parallel butterfly units would only matter if the
 * timing budget changed (02.Architecture/mapping/fft.md).
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
    output reg  bf_issue,
    output reg  bf_store,
    output reg  stage_next
);

localparam S_IDLE     = 3'd0,
           S_LOAD     = 3'd1,
           S_BF_ISSUE = 3'd2,
           S_BF_WAIT  = 3'd3,
           S_BF_STORE = 3'd4,
           S_STAGE    = 3'd5,
           S_DONE     = 3'd6;

reg [2:0] state, next_state;

/* state register */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state <= S_IDLE;
    else
        state <= next_state;
end

/* next-state (transition) logic -- pure decision, no side effects */
always @* begin
    case (state)
        S_IDLE:     next_state = start ? S_LOAD : S_IDLE;
        S_LOAD:     next_state = load_done ? S_BF_ISSUE : S_LOAD;
        S_BF_ISSUE: next_state = S_BF_WAIT;
        S_BF_WAIT:  next_state = bf_done ? S_BF_STORE : S_BF_WAIT;
        S_BF_STORE: next_state = bf_last ? S_STAGE : S_BF_ISSUE;
        S_STAGE:    next_state = stage_last ? S_DONE : S_BF_ISSUE;
        S_DONE:     next_state = S_IDLE;
        default:    next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    run_init   = (state == S_IDLE) && start;
    load_en    = (state == S_LOAD) && load_valid;
    bf_issue   = (state == S_BF_ISSUE);
    bf_store   = (state == S_BF_STORE);
    stage_next = (state == S_STAGE) && !stage_last;
end

endmodule
