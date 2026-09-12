/*
 * Module: cnn_seq_ctrl
 * Outer FSM of the CNN path: arm both spectrograms, wait until each has
 * a full frame, run the classifier over them, publish the class, arm
 * again.
 *
 * The two spectrograms do not finish on the same cycle -- the frame
 * buffer delivers one channel's block at a time, so one channel is
 * always a round ahead. Their done pulses are therefore latched outside
 * and this FSM waits on the latched pair.
 * Three separate always blocks: state register, next-state (transition)
 * logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module cnn_seq_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire specs_done,
    input  wire core_done,

    output reg  spec_start,
    output reg  clear_done,
    output reg  core_start,
    output reg  emit
);

localparam S_ARM      = 2'd0,
           S_COLLECT  = 2'd1,
           S_CLASSIFY = 2'd2,
           S_EMIT     = 2'd3;

reg [1:0] state, next_state;

/* state register */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        state <= S_ARM;
    else
        state <= next_state;
end

/* next-state (transition) logic -- pure decision, no side effects */
always @* begin
    case (state)
        S_ARM:      next_state = S_COLLECT;
        S_COLLECT:  next_state = specs_done ? S_CLASSIFY : S_COLLECT;
        S_CLASSIFY: next_state = core_done ? S_EMIT : S_CLASSIFY;
        S_EMIT:     next_state = S_ARM;
        default:    next_state = S_ARM;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    spec_start = (state == S_ARM);
    clear_done = (state == S_ARM);
    core_start = (state == S_COLLECT) && specs_done;
    emit       = (state == S_EMIT);
end

endmodule
