/*
 * Module: block_feeder_ctrl
 * FSM for block_feeder_datapath: accepts one block, then outputs one
 * sample per cycle, stalling while the consumer is busy.
 *
 * The stall keeps a new sample out of the decimator delay line during
 * a dot product.
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

/* next-state logic */
always @* begin
    case (state)
        S_IDLE:  next_state = valid_i ? S_SHIFT : S_IDLE;
        S_SHIFT: next_state = (!dst_busy && idx_last) ? S_IDLE : S_SHIFT;
        default: next_state = S_IDLE;
    endcase
end

/* output logic */
always @* begin
    ready_o  = (state == S_IDLE);
    do_load  = (state == S_IDLE) && valid_i;
    do_shift = (state == S_SHIFT) && !dst_busy;
end

endmodule
