/*
 * Module: block_feeder_ctrl
 * FSM for block_feeder_datapath: accept one sample block, then walk it
 * out one sample per cycle, stalling whenever the consumer is busy.
 * The stall matters -- the decimator computes a dot product across
 * several cycles and shifting a new sample into its delay line
 * mid-product would corrupt it.
 * Three separate always blocks: state register, next-state (transition)
 * logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module block_feeder_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire valid_i,
    input  wire dst_busy,
    input  wire idx_last,

    output reg  ready_o,
    output reg  do_load,
    output reg  do_shift
);

localparam S_IDLE  = 1'd0,
           S_SHIFT = 1'd1;

reg state, next_state;

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
        S_IDLE:  next_state = valid_i ? S_SHIFT : S_IDLE;
        S_SHIFT: next_state = (!dst_busy && idx_last) ? S_IDLE : S_SHIFT;
        default: next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- nothing else decided here */
always @* begin
    ready_o  = (state == S_IDLE);
    do_load  = (state == S_IDLE) && valid_i;
    do_shift = (state == S_SHIFT) && !dst_busy;
end

endmodule
