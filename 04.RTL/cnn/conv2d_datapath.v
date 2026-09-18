/*
 * Module: conv2d_datapath
 * 3x3 convolution, stride 1, zero padding 1, one filter.
 *
 * The 9 kernel positions go to mac_array one per cycle, CIN lanes
 * wide: one lane per input channel.
 * Lane reduction, bias and ReLU are adder and comparator logic outside
 * mac_array.
 * Q8.8 in and out, Q16.16 lane accumulators, rescaled to Q8.8 with
 * saturation after the lane reduction.
 * Image and feature map are memory ports. `img_addr` is the pixel
 * index shared by all channels; `img_data` carries the CIN channels of
 * that pixel. Image reads have one cycle of latency.
 * Control pulses come from conv2d_ctrl.
 * Status: implemented.
 */
module conv2d_datapath #(
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

    input  wire pixel_init,
    input  wire tap_init,
    input  wire tap_stream,
    input  wire do_reduce,
    input  wire do_bias,
    input  wire do_relu,
    input  wire do_write,
    input  wire pixel_step,

    output wire tap_last,
    output wire pixel_last,

    output wire [ADDR_BITS-1:0]                        img_addr,
    input  wire signed [CIN*WORD_BITS-1:0]             img_data,
    input  wire signed [KSIZE*KSIZE*CIN*WORD_BITS-1:0] kernel_in,
    input  wire signed [WORD_BITS-1:0]                 bias_in,

    output wire [ADDR_BITS-1:0]                        out_addr,
    output wire signed [WORD_BITS-1:0]                 out_word,
    output wire                                        out_we
);

integer oy, ox, ky, kx;
integer iy, ix;
integer c;

reg signed [WORD_BITS-1:0] biased;
reg signed [WORD_BITS-1:0] relu_result;
reg signed [ACC_BITS-1:0]  total;

always @* begin
    iy = oy + ky - 1;
    ix = ox + kx - 1;
end

wire in_bounds = (iy >= 0) && (iy < IMG_H) && (ix >= 0) && (ix < IMG_W);
wire is_first_tap = (ky == 0) && (kx == 0);

assign tap_last   = (kx == KSIZE-1) && (ky == KSIZE-1);
assign pixel_last = (oy == IMG_H-1) && (ox == IMG_W-1);

/* Address of this tap's source pixel. Taps outside the image are the
 * padding: the address is clamped to a valid pixel and the data is
 * forced to zero below. */
assign img_addr = in_bounds ? (iy * IMG_W + ix) : {ADDR_BITS{1'b0}};

assign out_addr = oy * IMG_W + ox;
assign out_word = relu_result;
assign out_we   = do_write;

/* Per-channel operands for one kernel position, one lane each.
 *
 * The image memory answers one cycle after the address, so the tap's
 * in-bounds flag, kernel weight and first-tap flag are delayed one
 * cycle to stay aligned with the pixel. */
reg signed [CIN*WORD_BITS-1:0] kernel_tap;
always @* begin
    for (c = 0; c < CIN; c = c + 1)
        kernel_tap[(c+1)*WORD_BITS-1 -: WORD_BITS] =
            kernel_in[(((ky*KSIZE+kx)*CIN+c)+1)*WORD_BITS-1 -: WORD_BITS];
end

reg signed [CIN*WORD_BITS-1:0] mac_b_r;
reg in_bounds_r, stream_r, first_r;

reg signed [CIN*WORD_BITS-1:0] mac_a;
always @* begin
    for (c = 0; c < CIN; c = c + 1)
        mac_a[(c+1)*WORD_BITS-1 -: WORD_BITS] =
            in_bounds_r ? img_data[(c+1)*WORD_BITS-1 -: WORD_BITS] : {WORD_BITS{1'b0}};
end

wire signed [CIN*WORD_BITS-1:0] mac_b = mac_b_r;
wire [CIN-1:0] mac_en    = {CIN{stream_r}};
wire [CIN-1:0] mac_clear = {CIN{first_r}};
wire signed [CIN*ACC_BITS-1:0] mac_acc;
wire [CIN-1:0] mac_valid;

mac_array #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .NUM_MACS(CIN)
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

/* Q16.16 total -> Q8.8, truncating and saturating. */
wire signed [ACC_BITS-FRAC_BITS-1:0] total_shifted = total >>> FRAC_BITS;
wire signed [WORD_BITS-1:0] Q88_MAX = {1'b0, {(WORD_BITS-1){1'b1}}};
wire signed [WORD_BITS-1:0] Q88_MIN = {1'b1, {(WORD_BITS-1){1'b0}}};
wire signed [WORD_BITS-1:0] total_q88 =
    (total_shifted > $signed(Q88_MAX)) ? Q88_MAX :
    (total_shifted < $signed(Q88_MIN)) ? Q88_MIN :
    total_shifted[WORD_BITS-1:0];

wire signed [WORD_BITS:0] bias_sum = total_q88 + bias_in;
wire signed [WORD_BITS-1:0] relu_out = (bias_sum[WORD_BITS] == bias_sum[WORD_BITS-1]) ?
    bias_sum[WORD_BITS-1:0] :                          // no overflow from the add
    (bias_sum[WORD_BITS] ? Q88_MIN : Q88_MAX);          // saturate on overflow
wire signed [WORD_BITS-1:0] relu_clamped = (biased[WORD_BITS-1]) ? {WORD_BITS{1'b0}} : biased;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        oy <= 0; ox <= 0; ky <= 0; kx <= 0;
        total <= {ACC_BITS{1'b0}};
        biased <= {WORD_BITS{1'b0}};
        relu_result <= {WORD_BITS{1'b0}};
        mac_b_r <= {CIN*WORD_BITS{1'b0}};
        in_bounds_r <= 1'b0;
        stream_r <= 1'b0;
        first_r  <= 1'b0;
    end else begin
        /* Delay by one cycle to align with the pixel read. */
        mac_b_r     <= kernel_tap;
        in_bounds_r <= in_bounds;
        stream_r    <= tap_stream;
        first_r     <= tap_stream && is_first_tap;

        if (pixel_init) begin
            oy <= 0;
            ox <= 0;
        end

        if (tap_init) begin
            ky <= 0;
            kx <= 0;
        end else if (tap_stream && !tap_last) begin
            if (kx == KSIZE-1) begin
                kx <= 0;
                ky <= ky + 1;
            end else begin
                kx <= kx + 1;
            end
        end

        if (do_reduce) begin
            total = {ACC_BITS{1'b0}};
            for (c = 0; c < CIN; c = c + 1)
                total = total + mac_acc[(c+1)*ACC_BITS-1 -: ACC_BITS];
        end

        if (do_bias)
            biased <= relu_out;

        if (do_relu)
            relu_result <= relu_clamped;

        if (pixel_step) begin
            if (ox == IMG_W-1) begin
                ox <= 0;
                oy <= oy + 1;
            end else begin
                ox <= ox + 1;
            end
        end
    end
end

endmodule
