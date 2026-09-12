/*
 * Module: magnitude_ctrl
 * FSM for magnitude_datapath: square both parts in the MAC lanes,
 * wait out the MAC pipeline, sum the lanes, then run the square root.
 * Three separate always blocks: state register, next-state
 * (transition) logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module magnitude_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire sqrt_done,

    output reg  busy,
    output reg  done,
    output reg  mac_issue,
    output reg  do_reduce,
    output reg  sqrt_start
);

localparam S_IDLE    = 3'd0,
           S_MAC     = 3'd1,
           S_DRAIN1  = 3'd2,
           S_DRAIN2  = 3'd3,
           S_REDUCE  = 3'd4,
           S_SQRT    = 3'd5,
           S_WAIT    = 3'd6,
           S_DONE    = 3'd7;

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
        S_IDLE:   next_state = start ? S_MAC : S_IDLE;
        S_MAC:    next_state = S_DRAIN1;
        S_DRAIN1: next_state = S_DRAIN2;
        S_DRAIN2: next_state = S_REDUCE;
        S_REDUCE: next_state = S_SQRT;
        S_SQRT:   next_state = S_WAIT;
        S_WAIT:   next_state = sqrt_done ? S_DONE : S_WAIT;
        S_DONE:   next_state = S_IDLE;
        default:  next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    mac_issue  = (state == S_MAC);
    do_reduce  = (state == S_REDUCE);
    sqrt_start = (state == S_SQRT);
end

endmodule
