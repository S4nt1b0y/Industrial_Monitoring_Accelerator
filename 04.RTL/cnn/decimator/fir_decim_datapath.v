/*
 * Module: fir_decim_datapath
 * Delay line, coefficient ROM, MAC lanes and output rescale for one
 * polyphase FIR decimation stage (/DECIM). Only computes an output
 * every DECIM-th input sample -- the taps in between are never
 * evaluated, which is what makes the /64 chain cost ~4.5 MACs per
 * input sample instead of 32.
 *
 * The TAPS-tap dot product is issued to ../common/mac_array.v in
 * batches of NUM_MACS lanes, so lane j accumulates taps j, j+NUM_MACS,
 * j+2*NUM_MACS ... and the lane totals are summed once at the end.
 * Reduction and the Q2.30 -> Q1.15 rescale are plain adder/saturate
 * logic outside the MAC bank.
 *
 * Every state decision comes from fir_decim_ctrl as a control pulse.
 * Status: implemented.
 */
module fir_decim_datapath #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,  // Q2.30
    parameter FRAC_BITS = 15,
    parameter TAPS      = 32,
    parameter NUM_MACS  = 4,
    parameter DECIM     = 8
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,

    input  wire batch_init,
    input  wire batch_step,
    input  wire do_reduce,
    input  wire do_output,

    output wire trigger,
    output wire batch_last,

    output reg  signed [WORD_BITS-1:0] sample_out,
    output reg                         sample_out_valid
);

integer i;

reg signed [WORD_BITS-1:0] delay [0:TAPS-1];
reg [7:0]  phase;
reg [7:0]  base;
reg signed [ACC_BITS-1:0] total;

/* an output is due once every DECIM input samples */
assign trigger    = sample_valid && (phase == DECIM-1);
assign batch_last = (base + NUM_MACS >= TAPS);

wire [4:0] coef_addr [0:NUM_MACS-1];
wire signed [WORD_BITS-1:0] coef_val [0:NUM_MACS-1];
reg  signed [NUM_MACS*WORD_BITS-1:0] mac_a, mac_b;

genvar g;
generate
    for (g = 0; g < NUM_MACS; g = g + 1) begin : coef_lane
        assign coef_addr[g] = base[4:0] + g[4:0];
        fir_coef_rom #(
            .WORD_BITS(WORD_BITS)
        ) u_coef_rom (
            .addr(coef_addr[g]),
            .coef(coef_val[g])
        );
    end
endgenerate

always @* begin
    for (i = 0; i < NUM_MACS; i = i + 1) begin
        mac_a[(i+1)*WORD_BITS-1 -: WORD_BITS] = delay[base + i];
        mac_b[(i+1)*WORD_BITS-1 -: WORD_BITS] = coef_val[i];
    end
end

wire [NUM_MACS-1:0] mac_en    = {NUM_MACS{batch_step}};
wire batch_first = (base == 8'd0);
wire [NUM_MACS-1:0] mac_clear = {NUM_MACS{batch_step && batch_first}};
wire signed [NUM_MACS*ACC_BITS-1:0] mac_acc;
wire [NUM_MACS-1:0] mac_valid;

mac_array #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .NUM_MACS(NUM_MACS)
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

/* Q2.30 -> Q1.15, saturating. Unity DC gain keeps a full-scale input
 * in range, so saturation should only ever bite on pathological data. */
wire signed [ACC_BITS-FRAC_BITS-1:0] total_shifted = total >>> FRAC_BITS;
wire signed [WORD_BITS-1:0] Q15_MAX = {1'b0, {(WORD_BITS-1){1'b1}}};
wire signed [WORD_BITS-1:0] Q15_MIN = {1'b1, {(WORD_BITS-1){1'b0}}};
wire signed [WORD_BITS-1:0] total_q15 =
    (total_shifted > $signed(Q15_MAX)) ? Q15_MAX :
    (total_shifted < $signed(Q15_MIN)) ? Q15_MIN :
    total_shifted[WORD_BITS-1:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < TAPS; i = i + 1)
            delay[i] <= {WORD_BITS{1'b0}};
        phase <= 8'd0;
        base  <= 8'd0;
        total <= {ACC_BITS{1'b0}};
        sample_out <= {WORD_BITS{1'b0}};
        sample_out_valid <= 1'b0;
    end else begin
        sample_out_valid <= 1'b0;

        if (sample_valid) begin
            for (i = TAPS-1; i > 0; i = i - 1)
                delay[i] <= delay[i-1];
            delay[0] <= sample_in;
            phase <= (phase == DECIM-1) ? 8'd0 : phase + 8'd1;
        end

        if (batch_init)
            base <= 8'd0;
        else if (batch_step && !batch_last)
            base <= base + NUM_MACS;

        if (do_reduce) begin
            total = {ACC_BITS{1'b0}};
            for (i = 0; i < NUM_MACS; i = i + 1)
                total = total + mac_acc[(i+1)*ACC_BITS-1 -: ACC_BITS];
        end

        if (do_output) begin
            sample_out <= total_q15;
            sample_out_valid <= 1'b1;
        end
    end
end

endmodule
