/*
 * Module: fft_butterfly
 * One radix-2 DIF butterfly, Q1.15 complex: A' = (A+B)/2 and
 * B' = ((A-B)/2) * W, with W = exp(-j*2*pi*k/64) given as the
 * (cos, sin) pair from twiddle_rom.
 *
 * The complex multiply is 4 real multiplies issued in one cycle to 4
 * mac_array lanes. The products are combined outside the array.
 * Both operands are halved before the add and subtract, because A-B
 * spans (-2,2) and does not fit Q1.15. This costs 1 LSB per stage and
 * scales the FFT output by 1/64.
 * Fixed 3-cycle latency: issue, 2 MAC pipeline stages, combine.
 * Status: implemented.
 */
module fft_butterfly #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,  // Q2.30 product
    parameter FRAC_BITS = 15
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        start,
    input  wire signed [WORD_BITS-1:0] ar,
    input  wire signed [WORD_BITS-1:0] ai,
    input  wire signed [WORD_BITS-1:0] br,
    input  wire signed [WORD_BITS-1:0] bi,
    input  wire signed [WORD_BITS-1:0] w_cos,
    input  wire signed [WORD_BITS-1:0] w_sin,

    output reg  signed [WORD_BITS-1:0] a_out_r,
    output reg  signed [WORD_BITS-1:0] a_out_i,
    output reg  signed [WORD_BITS-1:0] b_out_r,
    output reg  signed [WORD_BITS-1:0] b_out_i,
    output reg                         done
);

localparam NUM_LANES = 4;

wire signed [WORD_BITS-1:0] ar_h = ar >>> 1;
wire signed [WORD_BITS-1:0] ai_h = ai >>> 1;
wire signed [WORD_BITS-1:0] br_h = br >>> 1;
wire signed [WORD_BITS-1:0] bi_h = bi >>> 1;

wire signed [WORD_BITS-1:0] sum_r = ar_h + br_h;
wire signed [WORD_BITS-1:0] sum_i = ai_h + bi_h;
wire signed [WORD_BITS-1:0] dif_r = ar_h - br_h;
wire signed [WORD_BITS-1:0] dif_i = ai_h - bi_h;

/* lane0 = dr*cos, lane1 = di*sin, lane2 = di*cos, lane3 = dr*sin */
wire signed [NUM_LANES*WORD_BITS-1:0] mac_a = {dif_r, dif_i, dif_i, dif_r};
wire signed [NUM_LANES*WORD_BITS-1:0] mac_b = {w_sin, w_cos, w_sin, w_cos};

wire [NUM_LANES-1:0] mac_en    = {NUM_LANES{start}};
wire [NUM_LANES-1:0] mac_clear = {NUM_LANES{start}};
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

wire signed [ACC_BITS-1:0] p_dr_cos = mac_acc[1*ACC_BITS-1 -: ACC_BITS];
wire signed [ACC_BITS-1:0] p_di_sin = mac_acc[2*ACC_BITS-1 -: ACC_BITS];
wire signed [ACC_BITS-1:0] p_di_cos = mac_acc[3*ACC_BITS-1 -: ACC_BITS];
wire signed [ACC_BITS-1:0] p_dr_sin = mac_acc[4*ACC_BITS-1 -: ACC_BITS];

/* (dr + j*di) * (cos - j*sin) = (dr*cos + di*sin) + j*(di*cos - dr*sin) */
wire signed [ACC_BITS-1:0] prod_r = (p_dr_cos + p_di_sin) >>> FRAC_BITS;
wire signed [ACC_BITS-1:0] prod_i = (p_di_cos - p_dr_sin) >>> FRAC_BITS;

reg [2:0] pipe;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pipe    <= 3'b0;
        a_out_r <= {WORD_BITS{1'b0}};
        a_out_i <= {WORD_BITS{1'b0}};
        b_out_r <= {WORD_BITS{1'b0}};
        b_out_i <= {WORD_BITS{1'b0}};
        done    <= 1'b0;
    end else begin
        pipe <= {pipe[1:0], start};

        if (start) begin
            a_out_r <= sum_r;
            a_out_i <= sum_i;
        end

        if (pipe[2]) begin
            b_out_r <= prod_r[WORD_BITS-1:0];
            b_out_i <= prod_i[WORD_BITS-1:0];
        end

        done <= pipe[2];
    end
end

endmodule
