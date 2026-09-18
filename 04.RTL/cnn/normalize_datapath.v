/*
 * Module: normalize_datapath
 * Converts the Q1.15 spectrogram magnitudes to the Q8.8 activations
 * cnn_core uses, one element per pass: out = (mag*GAIN >> 8) - OFFSET,
 * saturating.
 *
 * This is the training z-score, (mag - mean)/std, in RTL units. GAIN
 * and OFFSET also absorb the 1/64 FFT scale and the prescale shift, and
 * are recomputed whenever the network is retrained (see
 * 04.RTL/cnn/README.md).
 * The multiply goes through mac_array (two cycles of latency).
 * Control pulses come from normalize_ctrl.
 * Status: implemented.
 */
module normalize_datapath #(
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

    input  wire elem_init,
    input  wire do_issue,
    input  wire do_write,
    input  wire elem_step,

    output wire elem_last,

    output wire [ADDR_BITS-1:0]        in_addr,
    input  wire [WORD_BITS-1:0]        in_data,

    output wire [ADDR_BITS-1:0]        out_addr,
    output wire signed [WORD_BITS-1:0] out_word,
    output wire                        out_we
);

integer idx;

assign elem_last = (idx == N_ELEMS-1);
assign in_addr   = idx[ADDR_BITS-1:0];
assign out_addr  = idx[ADDR_BITS-1:0];
assign out_we    = do_write;

/* The magnitude is unsigned and below 2^15, so a leading zero gives the
 * same value as a signed operand. */
wire signed [WORD_BITS-1:0] mag_signed = {1'b0, in_data[WORD_BITS-2:0]};

wire [0:0] mac_en    = do_issue;
wire [0:0] mac_clear = do_issue;
wire signed [ACC_BITS-1:0] mac_acc;
wire [0:0] mac_valid;

mac_array #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .NUM_MACS(1)
) u_mac_array (
    .clk(clk),
    .rst_n(rst_n),
    .en(mac_en),
    .clear(mac_clear),
    .a(mag_signed),
    .b(GAIN[WORD_BITS-1:0]),
    .acc(mac_acc),
    .valid(mac_valid)
);

wire signed [WORD_BITS-1:0] Q88_MAX = {1'b0, {(WORD_BITS-1){1'b1}}};
wire signed [WORD_BITS-1:0] Q88_MIN = {1'b1, {(WORD_BITS-1){1'b0}}};

wire signed [ACC_BITS-FRAC_BITS-1:0] scaled = mac_acc >>> FRAC_BITS;
wire signed [WORD_BITS-1:0] scaled_q88 =
    (scaled > $signed(Q88_MAX)) ? Q88_MAX :
    (scaled < $signed(Q88_MIN)) ? Q88_MIN :
    scaled[WORD_BITS-1:0];

wire signed [WORD_BITS:0] shifted = scaled_q88 - $signed(OFFSET[WORD_BITS-1:0]);
wire signed [WORD_BITS-1:0] result =
    (shifted[WORD_BITS] == shifted[WORD_BITS-1]) ?
        shifted[WORD_BITS-1:0] :                      // no overflow from the subtract
        (shifted[WORD_BITS] ? Q88_MIN : Q88_MAX);

assign out_word = result;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        idx <= 0;
    else if (elem_init)
        idx <= 0;
    else if (elem_step)
        idx <= idx + 1;
end

endmodule
