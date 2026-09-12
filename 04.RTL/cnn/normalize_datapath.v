/*
 * Module: normalize_datapath
 * Rewrites the spectrogram's Q1.15 magnitudes as the Q8.8 activations
 * the CNN core expects, one element per pass: out = (mag*GAIN >> 8) -
 * OFFSET, saturating.
 *
 * That single multiply-and-subtract is the training z-score expressed
 * in the RTL's units. The hardware spectrogram is not the reference one:
 * the FFT halves both butterfly operands at every stage (a factor of
 * 64), prescale.v shifts the decimated stream up, and the magnitude is
 * a Q1.15 integer rather than a float. Folding all of that into
 * (mag - mean)/std collapses to one gain and one offset, both constants.
 * Their values come from 03.Reference/tools/study_hw_quantization.py and
 * must be regenerated whenever the network is retrained.
 *
 * The multiply goes through mac_array like every other multiply in this
 * design, which is why there is a pass per element instead of a
 * combinational transform on the memory read path: a MAC lane answers
 * two cycles later, and conv2d reads its input memory combinationally.
 * Every state decision comes from normalize_ctrl as a one-cycle control
 * pulse; this module never decides what to do next on its own.
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

/* The magnitude is unsigned and never fills the top bit, so widening it
 * with a zero is the correct signed value for the MAC lane. */
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
