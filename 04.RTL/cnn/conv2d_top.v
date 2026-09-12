/*
 * Module: conv2d_top
 * Wires conv2d_ctrl (FSM) to conv2d_datapath (registers/MAC lanes) --
 * no logic of its own beyond that connection. Image in and feature map
 * out are memory ports; see conv2d_datapath's header.
 * Status: implemented.
 */
module conv2d_top #(
    parameter WORD_BITS = 16,
    parameter ACC_BITS  = 32,
    parameter FRAC_BITS = 8,
    parameter KSIZE     = 3,
    parameter CIN       = 1,
    parameter IMG_H     = 5,
    parameter IMG_W     = 5,
    parameter ADDR_BITS = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output wire busy,
    output wire done,

    output wire [ADDR_BITS-1:0]                        img_addr,
    input  wire signed [CIN*WORD_BITS-1:0]             img_data,
    input  wire signed [KSIZE*KSIZE*CIN*WORD_BITS-1:0] kernel_in,
    input  wire signed [WORD_BITS-1:0]                 bias_in,

    output wire [ADDR_BITS-1:0]                        out_addr,
    output wire signed [WORD_BITS-1:0]                 out_word,
    output wire                                        out_we
);

wire pixel_init, tap_init, tap_stream, do_reduce, do_bias, do_relu, do_write, pixel_step;
wire tap_last, pixel_last;

conv2d_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .tap_last(tap_last),
    .pixel_last(pixel_last),
    .busy(busy),
    .done(done),
    .pixel_init(pixel_init),
    .tap_init(tap_init),
    .tap_stream(tap_stream),
    .do_reduce(do_reduce),
    .do_bias(do_bias),
    .do_relu(do_relu),
    .do_write(do_write),
    .pixel_step(pixel_step)
);

conv2d_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .KSIZE(KSIZE),
    .CIN(CIN),
    .IMG_H(IMG_H),
    .IMG_W(IMG_W),
    .ADDR_BITS(ADDR_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .pixel_init(pixel_init),
    .tap_init(tap_init),
    .tap_stream(tap_stream),
    .do_reduce(do_reduce),
    .do_bias(do_bias),
    .do_relu(do_relu),
    .do_write(do_write),
    .pixel_step(pixel_step),
    .tap_last(tap_last),
    .pixel_last(pixel_last),
    .img_addr(img_addr),
    .img_data(img_data),
    .kernel_in(kernel_in),
    .bias_in(bias_in),
    .out_addr(out_addr),
    .out_word(out_word),
    .out_we(out_we)
);

endmodule
