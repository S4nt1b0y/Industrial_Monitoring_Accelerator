/*
 * Module: dense_datapath
 * Fully connected layer: N_IN activations to N_CLASSES logits.
 *
 * One MAC lane per class: all dot products advance together, one
 * activation per cycle, N_IN cycles in total.
 * Rescale to Q8.8, bias and saturation are adder and comparator logic
 * outside mac_array.
 * Q8.8 in and out, Q16.16 lane accumulators.
 * Activations and weights are memory ports. `in_addr` walks 0..N_IN-1;
 * the activation memory holds the flatten order filter*256 + y*16 + x.
 * `w_data` carries the N_CLASSES weights for that index.
 * Both memories have one cycle of read latency, so activation and
 * weight arrive aligned; the MAC enable is delayed one cycle.
 * Control pulses come from dense_ctrl.
 * Status: implemented.
 */
module dense_datapath #(
    parameter WORD_BITS  = 16,
    parameter ACC_BITS   = 32,
    parameter FRAC_BITS  = 8,
    parameter N_IN       = 8,
    parameter N_CLASSES  = 4,
    parameter ADDR_BITS  = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire tap_init,
    input  wire tap_stream,
    input  wire do_bias,

    output wire tap_last,

    output wire [ADDR_BITS-1:0]                   in_addr,
    input  wire signed [WORD_BITS-1:0]            in_data,
    output wire [ADDR_BITS-1:0]                   w_addr,
    input  wire signed [N_CLASSES*WORD_BITS-1:0]  w_data,
    input  wire signed [N_CLASSES*WORD_BITS-1:0]  bias_in,

    output reg  signed [N_CLASSES*WORD_BITS-1:0]  logits
);

integer idx;

reg stream_r;
reg first_r;

assign tap_last = (idx == N_IN-1);

assign in_addr = idx[ADDR_BITS-1:0];
assign w_addr  = idx[ADDR_BITS-1:0];

/* One lane per class: same activation, different weight. */
wire [N_CLASSES-1:0] mac_en    = {N_CLASSES{stream_r}};
wire [N_CLASSES-1:0] mac_clear = {N_CLASSES{first_r}};
wire signed [N_CLASSES*WORD_BITS-1:0] mac_a = {N_CLASSES{in_data}};
wire signed [N_CLASSES*ACC_BITS-1:0]  mac_acc;
wire [N_CLASSES-1:0] mac_valid;

mac_array #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .NUM_MACS(N_CLASSES)
) u_mac_array (
    .clk(clk),
    .rst_n(rst_n),
    .en(mac_en),
    .clear(mac_clear),
    .a(mac_a),
    .b(w_data),
    .acc(mac_acc),
    .valid(mac_valid)
);

wire signed [WORD_BITS-1:0] Q88_MAX = {1'b0, {(WORD_BITS-1){1'b1}}};
wire signed [WORD_BITS-1:0] Q88_MIN = {1'b1, {(WORD_BITS-1){1'b0}}};

wire signed [N_CLASSES*WORD_BITS-1:0] logits_next;

/* Q16.16 lane sum -> Q8.8, truncating and saturating, then bias. */
genvar gc;
generate
    for (gc = 0; gc < N_CLASSES; gc = gc + 1) begin : cls
        /* A part-select of a signed vector is unsigned in Verilog;
         * $signed keeps the bias add signed. */
        wire signed [ACC_BITS-1:0] total = $signed(mac_acc[(gc+1)*ACC_BITS-1 -: ACC_BITS]);
        wire signed [ACC_BITS-FRAC_BITS-1:0] shifted = total >>> FRAC_BITS;
        wire signed [WORD_BITS-1:0] total_q88 =
            (shifted > $signed(Q88_MAX)) ? Q88_MAX :
            (shifted < $signed(Q88_MIN)) ? Q88_MIN :
            shifted[WORD_BITS-1:0];
        wire signed [WORD_BITS:0] bias_sum =
            total_q88 + $signed(bias_in[(gc+1)*WORD_BITS-1 -: WORD_BITS]);
        assign logits_next[(gc+1)*WORD_BITS-1 -: WORD_BITS] =
            (bias_sum[WORD_BITS] == bias_sum[WORD_BITS-1]) ?
                bias_sum[WORD_BITS-1:0] :               // no overflow from the add
                (bias_sum[WORD_BITS] ? Q88_MIN : Q88_MAX);
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        idx      <= 0;
        stream_r <= 1'b0;
        first_r  <= 1'b0;
        logits   <= {N_CLASSES*WORD_BITS{1'b0}};
    end else begin
        if (tap_init)
            idx <= 0;
        else if (tap_stream && !tap_last)
            idx <= idx + 1;

        stream_r <= tap_stream;
        first_r  <= tap_stream && (idx == 0);

        if (do_bias)
            logits <= logits_next;
    end
end

endmodule
