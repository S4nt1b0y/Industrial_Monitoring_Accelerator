/*
 * Module: fir_decim_ctrl
 * FSM for fir_decim_datapath: idle until the datapath signals an
 * output is due (one every DECIM input samples), then issue the tap
 * batches, wait out the MAC pipeline, reduce the lanes and emit.
 * Three separate always blocks: state register, next-state
 * (transition) logic, and output control-pulse logic -- never mixed.
 *
 * Assumes input samples arrive far apart relative to the compute time
 * (here ~12 cycles against thousands between samples at 25.6 kHz), so
 * the delay line is never shifted mid-dot-product. `busy` is exported
 * so a faster source can check.
 * Status: implemented.
 */
module fir_decim_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire trigger,
    input  wire batch_last,

    output reg  busy,
    output reg  batch_init,
    output reg  batch_step,
    output reg  do_reduce,
    output reg  do_output
);

localparam S_IDLE       = 3'd0,
           S_BATCH_INIT = 3'd1,
           S_BATCH      = 3'd2,
           S_DRAIN1     = 3'd3,
           S_DRAIN2     = 3'd4,
           S_REDUCE     = 3'd5,
           S_OUTPUT     = 3'd6;

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
        S_IDLE:       next_state = trigger ? S_BATCH_INIT : S_IDLE;
        S_BATCH_INIT: next_state = S_BATCH;
        S_BATCH:      next_state = batch_last ? S_DRAIN1 : S_BATCH;
        S_DRAIN1:     next_state = S_DRAIN2;
        S_DRAIN2:     next_state = S_REDUCE;
        S_REDUCE:     next_state = S_OUTPUT;
        S_OUTPUT:     next_state = S_IDLE;
        default:      next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy       = (state != S_IDLE);
    batch_init = (state == S_BATCH_INIT);
    batch_step = (state == S_BATCH);
    do_reduce  = (state == S_REDUCE);
    do_output  = (state == S_OUTPUT);
end

endmodule
