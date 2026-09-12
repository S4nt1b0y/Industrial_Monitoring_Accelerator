/*
 * Module: maxpool
 * Wires maxpool_ctrl (FSM) to maxpool_datapath (counters, comparator) --
 * no logic of its own beyond that connection. Feature map in and pooled
 * map out are memory ports; see maxpool_datapath's header.
 * Status: implemented.
 */
module maxpool #(
    parameter WORD_BITS = 16,
    parameter POOL      = 2,
    parameter IMG_H     = 4,
    parameter IMG_W     = 4,
    parameter ADDR_BITS = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output wire busy,
    output wire done,

    output wire [ADDR_BITS-1:0]        in_addr,
    input  wire signed [WORD_BITS-1:0] in_data,

    output wire [ADDR_BITS-1:0]        out_addr,
    output wire signed [WORD_BITS-1:0] out_word,
    output wire                        out_we
);

wire pixel_init, tap_init, tap_stream, do_write, pixel_step;
wire tap_last, pixel_last;

maxpool_ctrl u_ctrl (
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
    .do_write(do_write),
    .pixel_step(pixel_step)
);

maxpool_datapath #(
    .WORD_BITS(WORD_BITS),
    .POOL(POOL),
    .IMG_H(IMG_H),
    .IMG_W(IMG_W),
    .ADDR_BITS(ADDR_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .pixel_init(pixel_init),
    .tap_init(tap_init),
    .tap_stream(tap_stream),
    .do_write(do_write),
    .pixel_step(pixel_step),
    .tap_last(tap_last),
    .pixel_last(pixel_last),
    .in_addr(in_addr),
    .in_data(in_data),
    .out_addr(out_addr),
    .out_word(out_word),
    .out_we(out_we)
);

endmodule
