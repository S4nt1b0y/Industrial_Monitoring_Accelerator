/*
 * Module: maxpool_ctrl
 * FSM for maxpool_datapath: one output pixel at a time. For each pixel
 * the POOL*POOL window positions go to the comparator one per cycle,
 * then the maximum is written.
 *
 * S_LAG waits one cycle for the input memory to return the last tap.
 * Status: implemented.
 */
module maxpool_ctrl (
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
    output reg  do_write,
    output reg  pixel_step
);

localparam S_IDLE       = 3'd0,
           S_PIXEL_INIT = 3'd1,
           S_TAP_INIT   = 3'd2,
           S_TAP_STREAM = 3'd3,
           S_LAG        = 3'd4,
           S_WRITE      = 3'd5,
           S_PIXEL_NEXT = 3'd6,
           S_DONE       = 3'd7;

reg [2:0] state, next_state;

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
        S_IDLE:       next_state = start ? S_PIXEL_INIT : S_IDLE;
        S_PIXEL_INIT: next_state = S_TAP_INIT;
        S_TAP_INIT:   next_state = S_TAP_STREAM;
        S_TAP_STREAM: next_state = tap_last ? S_LAG : S_TAP_STREAM;
        S_LAG:        next_state = S_WRITE;
        S_WRITE:      next_state = pixel_last ? S_DONE : S_PIXEL_NEXT;
        S_PIXEL_NEXT: next_state = S_TAP_INIT;
        S_DONE:       next_state = S_IDLE;
        default:      next_state = S_IDLE;
    endcase
end

/* output logic */
always @* begin
    busy       = (state != S_IDLE);
    done       = (state == S_DONE);
    pixel_init = (state == S_PIXEL_INIT);
    tap_init   = (state == S_TAP_INIT);
    tap_stream = (state == S_TAP_STREAM);
    do_write   = (state == S_WRITE);
    pixel_step = (state == S_PIXEL_NEXT);
end

endmodule
