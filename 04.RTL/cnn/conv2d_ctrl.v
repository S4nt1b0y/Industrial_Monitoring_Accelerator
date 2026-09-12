/*
 * Module: conv2d_ctrl
 * FSM for conv2d_datapath: one output pixel at a time; inside each
 * pixel the 9 kernel positions stream one per cycle into the MAC
 * array (all CIN channels in parallel), then two drain cycles for the
 * MAC pipeline, then reduce/bias/ReLU/write. Three separate always
 * blocks: state register, next-state (transition) logic, and output
 * control-pulse logic -- never mixed.
 * Status: implemented.
 */
module conv2d_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire tap_last,
    input  wire pixel_last,

    output reg  busy,
    output reg  done,
    output reg  pixel_init,
    output reg  tap_init,
    output reg  tap_stream,
    output reg  do_reduce,
    output reg  do_bias,
    output reg  do_relu,
    output reg  do_write,
    output reg  pixel_step
);

localparam S_IDLE       = 4'd0,
           S_PIXEL_INIT = 4'd1,
           S_TAP_INIT   = 4'd2,
           S_TAP_STREAM = 4'd3,
           S_DRAIN1     = 4'd4,
           S_DRAIN2     = 4'd5,
           S_REDUCE     = 4'd6,
           S_BIAS       = 4'd7,
           S_RELU       = 4'd8,
           S_WRITE      = 4'd9,
           S_PIXEL_NEXT = 4'd10,
           S_DONE       = 4'd11;

reg [3:0] state, next_state;

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
        S_IDLE:       next_state = start ? S_PIXEL_INIT : S_IDLE;
        S_PIXEL_INIT: next_state = S_TAP_INIT;
        S_TAP_INIT:   next_state = S_TAP_STREAM;
        S_TAP_STREAM: next_state = tap_last ? S_DRAIN1 : S_TAP_STREAM;
        S_DRAIN1:     next_state = S_DRAIN2;
        S_DRAIN2:     next_state = S_REDUCE;
        S_REDUCE:     next_state = S_BIAS;
        S_BIAS:       next_state = S_RELU;
        S_RELU:       next_state = S_WRITE;
        S_WRITE:      next_state = pixel_last ? S_DONE : S_PIXEL_NEXT;
        S_PIXEL_NEXT: next_state = S_TAP_INIT;
        S_DONE:       next_state = S_IDLE;
        default:      next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    pixel_init = (state == S_PIXEL_INIT);
    tap_init   = (state == S_TAP_INIT);
    tap_stream = (state == S_TAP_STREAM);
    do_reduce  = (state == S_REDUCE);
    do_bias    = (state == S_BIAS);
    do_relu    = (state == S_RELU);
    do_write   = (state == S_WRITE);
    pixel_step = (state == S_PIXEL_NEXT);
end

endmodule
