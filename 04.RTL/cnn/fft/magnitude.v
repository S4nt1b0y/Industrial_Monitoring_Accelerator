/*
 * Module: magnitude
 * |X| = sqrt(re^2 + im^2) for one FFT bin, Q1.15 in and out. Wires
 * magnitude_ctrl (FSM) to magnitude_datapath (MAC lanes + isqrt); no
 * logic of its own beyond that connection.
 * Status: implemented.
 */
module magnitude #(
    parameter WORD_BITS = 16,
    parameter ACC_BITS  = 32
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        start,
    input  wire signed [WORD_BITS-1:0] re,
    input  wire signed [WORD_BITS-1:0] im,

    output wire                        busy,
    output wire                        done,
    output wire [WORD_BITS-1:0]        mag
);

wire mac_issue, do_reduce, sqrt_start, sqrt_done;

magnitude_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .sqrt_done(sqrt_done),
    .busy(busy),
    .done(done),
    .mac_issue(mac_issue),
    .do_reduce(do_reduce),
    .sqrt_start(sqrt_start)
);

magnitude_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .mac_issue(mac_issue),
    .do_reduce(do_reduce),
    .sqrt_start(sqrt_start),
    .re(re),
    .im(im),
    .sqrt_done(sqrt_done),
    .mag(mag)
);

endmodule
