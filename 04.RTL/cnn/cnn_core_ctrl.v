/*
 * Module: cnn_core_ctrl
 * FSM that sequences one classification: normalize each input channel,
 * then run convolution and pooling once per filter, then the dense layer
 * and argmax.
 *
 * One convolution unit is reused across the 8 filters rather than eight
 * running in parallel. The cycle budget is not the constraint here -- a
 * full pass is a few hundred thousand cycles against the tens of
 * millions the UART takes to deliver the samples -- while DSP slices
 * are, so serialising trades a resource that is scarce for one that is
 * not.
 * Three separate always blocks: state register, next-state (transition)
 * logic, and output control-pulse logic -- never mixed.
 * Status: implemented.
 */
module cnn_core_ctrl (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire norm_done,
    input  wire conv_done,
    input  wire pool_done,
    input  wire dense_done,
    input  wire chan_last,
    input  wire filter_last,

    output reg  busy,
    output reg  done,

    output reg  chan_init,
    output reg  chan_step,
    output reg  filter_init,
    output reg  filter_step,

    output reg  norm_start,
    output reg  conv_start,
    output reg  pool_start,
    output reg  dense_start,
    output reg  latch_class
);

localparam S_IDLE        = 4'd0,
           S_CHAN_INIT   = 4'd1,
           S_NORM_START  = 4'd2,
           S_NORM_WAIT   = 4'd3,
           S_CHAN_NEXT   = 4'd4,
           S_FILTER_INIT = 4'd5,
           S_CONV_START  = 4'd6,
           S_CONV_WAIT   = 4'd7,
           S_POOL_START  = 4'd8,
           S_POOL_WAIT   = 4'd9,
           S_FILTER_NEXT = 4'd10,
           S_DENSE_START = 4'd11,
           S_DENSE_WAIT  = 4'd12,
           S_LATCH       = 4'd13,
           S_DONE        = 4'd14;

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
        S_IDLE:        next_state = start ? S_CHAN_INIT : S_IDLE;
        S_CHAN_INIT:   next_state = S_NORM_START;
        S_NORM_START:  next_state = S_NORM_WAIT;
        S_NORM_WAIT:   next_state = norm_done ?
                                    (chan_last ? S_FILTER_INIT : S_CHAN_NEXT) :
                                    S_NORM_WAIT;
        S_CHAN_NEXT:   next_state = S_NORM_START;
        S_FILTER_INIT: next_state = S_CONV_START;
        S_CONV_START:  next_state = S_CONV_WAIT;
        S_CONV_WAIT:   next_state = conv_done ? S_POOL_START : S_CONV_WAIT;
        S_POOL_START:  next_state = S_POOL_WAIT;
        S_POOL_WAIT:   next_state = pool_done ?
                                    (filter_last ? S_DENSE_START : S_FILTER_NEXT) :
                                    S_POOL_WAIT;
        S_FILTER_NEXT: next_state = S_CONV_START;
        S_DENSE_START: next_state = S_DENSE_WAIT;
        S_DENSE_WAIT:  next_state = dense_done ? S_LATCH : S_DENSE_WAIT;
        S_LATCH:       next_state = S_DONE;
        S_DONE:        next_state = S_IDLE;
        default:       next_state = S_IDLE;
    endcase
end

/* output control-pulse logic -- one pulse per state, nothing else */
always @* begin
    busy        = (state != S_IDLE);
    done        = (state == S_DONE);
    chan_init   = (state == S_CHAN_INIT);
    chan_step   = (state == S_CHAN_NEXT);
    filter_init = (state == S_FILTER_INIT);
    filter_step = (state == S_FILTER_NEXT);
    norm_start  = (state == S_NORM_START);
    conv_start  = (state == S_CONV_START);
    pool_start  = (state == S_POOL_START);
    dense_start = (state == S_DENSE_START);
    latch_class = (state == S_LATCH);
end

endmodule
