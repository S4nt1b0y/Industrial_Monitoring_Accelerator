/*
 * Module: normalize_ctrl
 * FSM for normalize_datapath, one element per pass: fetch from the
 * spectrogram memory, multiply in the MAC lane, wait for its two
 * pipeline stages, write, advance. Done after the last element.
 *
 * S_FETCH is the cycle in which the spectrogram memory returns the
 * data.
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

localparam S_IDLE      = 4'd0,
           S_ELEM_INIT = 4'd1,
           S_FETCH     = 4'd2,
           S_ISSUE     = 4'd3,
           S_DRAIN1    = 4'd4,
           S_DRAIN2    = 4'd5,
           S_WRITE     = 4'd6,
           S_ELEM_NEXT = 4'd7,
           S_DONE      = 4'd8;

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
        S_IDLE:      next_state = start ? S_ELEM_INIT : S_IDLE;
        S_ELEM_INIT: next_state = S_FETCH;
        S_FETCH:     next_state = S_ISSUE;
        S_ISSUE:     next_state = S_DRAIN1;
        S_DRAIN1:    next_state = S_DRAIN2;
        S_DRAIN2:    next_state = S_WRITE;
        S_WRITE:     next_state = elem_last ? S_DONE : S_ELEM_NEXT;
        S_ELEM_NEXT: next_state = S_FETCH;
        S_DONE:      next_state = S_IDLE;
        default:     next_state = S_IDLE;
    endcase
end

/* output logic */
always @* begin
    busy      = (state != S_IDLE);
    done      = (state == S_DONE);
    elem_init = (state == S_ELEM_INIT);
    do_issue  = (state == S_ISSUE);
    do_write  = (state == S_WRITE);
    elem_step = (state == S_ELEM_NEXT);
end

endmodule
