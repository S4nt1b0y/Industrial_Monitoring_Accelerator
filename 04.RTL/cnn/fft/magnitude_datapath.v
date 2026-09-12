/*
 * Module: magnitude_datapath
 * |X| = sqrt(re^2 + im^2) for one FFT bin. The two squares go to
 * ../common/mac_array.v as two parallel lanes (every multiply in this
 * project goes through the MAC bank); summing the lanes is plain adder
 * logic, and the square root is the multiplier-free isqrt.
 *
 * Widths: re, im are Q1.15, so each square is Q2.30 and their sum is
 * below 2.0, which still fits a signed 32-bit Q2.30 word. isqrt of a
 * Q2.30 value returns Q1.15 directly.
 *
 * Every state decision comes from magnitude_ctrl as a control pulse.
 * Status: implemented.
 */
module magnitude_datapath #(
    parameter WORD_BITS = 16,
    parameter ACC_BITS  = 32
) (
    input  wire clk,
    input  wire rst_n,

    input  wire mac_issue,
    input  wire do_reduce,
    input  wire sqrt_start,

    input  wire signed [WORD_BITS-1:0] re,
    input  wire signed [WORD_BITS-1:0] im,

    output wire                        sqrt_done,
    output wire [WORD_BITS-1:0]        mag
);

localparam NUM_LANES = 2;

integer i;

wire signed [NUM_LANES*WORD_BITS-1:0] mac_a = {im, re};
wire signed [NUM_LANES*WORD_BITS-1:0] mac_b = {im, re};
wire [NUM_LANES-1:0] mac_en    = {NUM_LANES{mac_issue}};
wire [NUM_LANES-1:0] mac_clear = {NUM_LANES{mac_issue}};
wire signed [NUM_LANES*ACC_BITS-1:0] mac_acc;
wire [NUM_LANES-1:0] mac_valid;

mac_array #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .NUM_MACS(NUM_LANES)
) u_mac_array (
    .clk(clk),
    .rst_n(rst_n),
    .en(mac_en),
    .clear(mac_clear),
    .a(mac_a),
    .b(mac_b),
    .acc(mac_acc),
    .valid(mac_valid)
);

reg [ACC_BITS-1:0] sum_sq;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        sum_sq <= {ACC_BITS{1'b0}};
    else if (do_reduce)
        sum_sq <= mac_acc[1*ACC_BITS-1 -: ACC_BITS] + mac_acc[2*ACC_BITS-1 -: ACC_BITS];
end

isqrt #(
    .WORD_BITS(WORD_BITS)
) u_isqrt (
    .clk(clk),
    .rst_n(rst_n),
    .start(sqrt_start),
    .value(sum_sq),
    .busy(),
    .done(sqrt_done),
    .root(mag)
);

endmodule
