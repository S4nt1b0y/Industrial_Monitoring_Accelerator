/*
 * Module: normalize_ctrl
 * FSM for normalize_datapath: one element per pass -- issue the multiply
 * to the MAC lane, wait out its two pipeline stages, write the result,
 * advance. Runs until the last element, then done.
 * Three separate always blocks: state register, next-state (transition)
 * logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module normalize_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire elem_last,

    output reg  busy,
    output reg  done,
    output reg  elem_init,
    output reg  do_issue,
    output reg  do_write,
    output reg  elem_step
);

localparam S_IDLE      = 3'd0,
           S_ELEM_INIT = 3'd1,
           S_ISSUE     = 3'd2,
           S_DRAIN1    = 3'd3,
           S_DRAIN2    = 3'd4,
           S_WRITE     = 3'd5,
           S_ELEM_NEXT = 3'd6,
           S_DONE      = 3'd7;

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
        S_IDLE:      next_state = start ? S_ELEM_INIT : S_IDLE;
        S_ELEM_INIT: next_state = S_ISSUE;
        S_ISSUE:     next_state = S_DRAIN1;
        S_DRAIN1:    next_state = S_DRAIN2;
        S_DRAIN2:    next_state = S_WRITE;
        S_WRITE:     next_state = elem_last ? S_DONE : S_ELEM_NEXT;
        S_ELEM_NEXT: next_state = S_ISSUE;
        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy      = (state != S_IDLE);
    done      = (state == S_DONE);
    elem_init = (state == S_ELEM_INIT);
    do_issue  = (state == S_ISSUE);
    do_write  = (state == S_WRITE);
    elem_step = (state == S_ELEM_NEXT);
end

endmodule
