/*
 * Module: dense_datapath
 * Fully connected layer: N_IN activations to N_CLASSES logits. One MAC
 * lane per class, so all N_CLASSES dot products advance together on one
 * activation per cycle -- N_IN cycles total instead of N_CLASSES*N_IN.
 * The rescale to Q8.8, the bias add and the saturation are plain
 * adder/comparator logic outside the DSP-oriented MAC array.
 * Every state decision comes from dense_ctrl as a one-cycle control
 * pulse; this module never decides what to do next on its own.
 * Q8.8 in/out, Q16.16 per-lane accumulator.
 *
 * Activations and weights are memory ports. `in_addr` walks 0..N_IN-1,
 * so whoever fills the activation memory owns the flatten order -- for
 * this CNN that is filter*256 + y*16 + x, C order, matching
 * `pooled.reshape(-1)` in 03.Reference/cnn/reference.py. `w_data`
 * carries all N_CLASSES weights for that same index, so the caller
 * wires one weight memory per lane.
 *
 * The two memories do not answer in the same cycle: ram.v reads
 * combinationally and weight_rom.v reads one cycle later. The
 * activation is therefore held one cycle in `act_r` to meet its weight,
 * and the MAC enable runs one cycle behind the address counter.
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

reg signed [WORD_BITS-1:0] act_r;
reg                        stream_r;
reg                        first_r;

assign tap_last = (idx == N_IN-1);

assign in_addr = idx[ADDR_BITS-1:0];
assign w_addr  = idx[ADDR_BITS-1:0];

/* One lane per class: same activation, different weight. */
wire [N_CLASSES-1:0] mac_en    = {N_CLASSES{stream_r}};
wire [N_CLASSES-1:0] mac_clear = {N_CLASSES{first_r}};
wire signed [N_CLASSES*WORD_BITS-1:0] mac_a = {N_CLASSES{act_r}};
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

/* Q16.16 lane total -> Q8.8, saturating, then bias. Truncating (no
 * round-to-nearest), the same choice conv2d_datapath makes. */
genvar gc;
generate
    for (gc = 0; gc < N_CLASSES; gc = gc + 1) begin : cls
        /* $signed on every part-select: a slice of a signed vector is
         * unsigned in Verilog, and an unsigned bias would turn the add
         * below into unsigned arithmetic and trip the overflow guard. */
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
        act_r    <= {WORD_BITS{1'b0}};
        stream_r <= 1'b0;
        first_r  <= 1'b0;
        logits   <= {N_CLASSES*WORD_BITS{1'b0}};
    end else begin
        if (tap_init)
            idx <= 0;
        else if (tap_stream && !tap_last)
            idx <= idx + 1;

        /* Hold the activation one cycle so it lands with its weight. */
        act_r    <= in_data;
        stream_r <= tap_stream;
        first_r  <= tap_stream && (idx == 0);

        if (do_bias)
            logits <= logits_next;
    end
end

endmodule
