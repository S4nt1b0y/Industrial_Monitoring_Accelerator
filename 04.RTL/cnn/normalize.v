/*
 * Module: normalize
 * Wires normalize_ctrl (FSM) to normalize_datapath (counter, MAC lane,
 * rescale) -- no logic of its own beyond that connection. Bridges the
 * Q1.15 spectrogram to the Q8.8 activations the CNN core reads; see
 * normalize_datapath's header for where GAIN and OFFSET come from.
 * Status: implemented.
 */
module normalize #(
    parameter WORD_BITS = 16,
    parameter ACC_BITS  = 32,
    parameter FRAC_BITS = 8,
    parameter N_ELEMS   = 1024,
    parameter ADDR_BITS = 12,
    parameter GAIN      = 287,  // Q8.8
    parameter OFFSET    = 116   // Q8.8
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    output wire busy,
    output wire done,

    output wire [ADDR_BITS-1:0]        in_addr,
    input  wire [WORD_BITS-1:0]        in_data,

    output wire [ADDR_BITS-1:0]        out_addr,
    output wire signed [WORD_BITS-1:0] out_word,
    output wire                        out_we
);

wire elem_init, do_issue, do_write, elem_step;
wire elem_last;

normalize_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .elem_last(elem_last),
    .busy(busy),
    .done(done),
    .elem_init(elem_init),
    .do_issue(do_issue),
    .do_write(do_write),
    .elem_step(elem_step)
);

normalize_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .N_ELEMS(N_ELEMS),
    .ADDR_BITS(ADDR_BITS),
    .GAIN(GAIN),
    .OFFSET(OFFSET)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .elem_init(elem_init),
    .do_issue(do_issue),
    .do_write(do_write),
    .elem_step(elem_step),
    .elem_last(elem_last),
    .in_addr(in_addr),
    .in_data(in_data),
    .out_addr(out_addr),
    .out_word(out_word),
    .out_we(out_we)
);

endmodule
