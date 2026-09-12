/*
 * Module: dense_ctrl
 * FSM for dense_datapath: one activation per cycle for N_IN cycles,
 * then three drain states before the totals are read. Three, not two:
 * one covers the weight memory's registered read, which puts the MAC
 * enable a cycle behind the address counter, and two more flush the MAC
 * multiply/accumulate pipeline.
 * Three separate always blocks: state register, next-state (transition)
 * logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module dense_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire tap_last,

    output reg  busy,
    output reg  done,
    output reg  tap_init,
    output reg  tap_stream,
    output reg  do_bias
);

localparam S_IDLE       = 3'd0,
           S_TAP_INIT   = 3'd1,
           S_TAP_STREAM = 3'd2,
           S_DRAIN1     = 3'd3,
           S_DRAIN2     = 3'd4,
           S_DRAIN3     = 3'd5,
           S_BIAS       = 3'd6,
           S_DONE       = 3'd7;

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
        S_IDLE:       next_state = start ? S_TAP_INIT : S_IDLE;
        S_TAP_INIT:   next_state = S_TAP_STREAM;
        S_TAP_STREAM: next_state = tap_last ? S_DRAIN1 : S_TAP_STREAM;
        S_DRAIN1:     next_state = S_DRAIN2;
        S_DRAIN2:     next_state = S_DRAIN3;
        S_DRAIN3:     next_state = S_BIAS;
        S_BIAS:       next_state = S_DONE;
        S_DONE:       next_state = S_IDLE;
        default:      next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    tap_init   = (state == S_TAP_INIT);
    tap_stream = (state == S_TAP_STREAM);
    do_bias    = (state == S_BIAS);
end

endmodule
