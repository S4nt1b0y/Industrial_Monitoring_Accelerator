/*
 * Module: dense
 * Wires dense_ctrl (FSM) to dense_datapath (counter, MAC lanes,
 * rescale/bias) -- no logic of its own beyond that connection.
 * Activations and weights are memory ports; see dense_datapath's header
 * for the flatten order and the read-timing note.
 * Status: implemented.
 */
module dense #(
    parameter WORD_BITS  = 16,
    parameter ACC_BITS   = 32,
    parameter FRAC_BITS  = 8,
    parameter N_IN       = 8,
    parameter N_CLASSES  = 4,
    parameter ADDR_BITS  = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output wire busy,
    output wire done,

    output wire [ADDR_BITS-1:0]                   in_addr,
    input  wire signed [WORD_BITS-1:0]            in_data,
    output wire [ADDR_BITS-1:0]                   w_addr,
    input  wire signed [N_CLASSES*WORD_BITS-1:0]  w_data,
    input  wire signed [N_CLASSES*WORD_BITS-1:0]  bias_in,

    output wire signed [N_CLASSES*WORD_BITS-1:0]  logits
);

wire tap_init, tap_stream, do_bias;
wire tap_last;

dense_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .tap_last(tap_last),
    .busy(busy),
    .done(done),
    .tap_init(tap_init),
    .tap_stream(tap_stream),
    .do_bias(do_bias)
);

dense_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .N_IN(N_IN),
    .N_CLASSES(N_CLASSES),
    .ADDR_BITS(ADDR_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .tap_init(tap_init),
    .tap_stream(tap_stream),
    .do_bias(do_bias),
    .tap_last(tap_last),
    .in_addr(in_addr),
    .in_data(in_data),
    .w_addr(w_addr),
    .w_data(w_data),
    .bias_in(bias_in),
    .logits(logits)
);

endmodule
