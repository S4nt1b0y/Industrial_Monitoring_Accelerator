/*
 * Testbench: dense
 * Fully connected layer with N_IN=8 and 4 classes, every expected logit
 * worked out by hand in Q8.8 (1.0 = 256). Two runs with different
 * weights, so the second also proves the layer restarts cleanly and
 * that the MAC accumulators really clear instead of carrying the first
 * run's totals.
 * The activations mix positive, negative and zero, and the weights
 * include a one-hot row -- the case that catches an off-by-one in the
 * address counter, since only one index contributes.
 * argmax is wired to the logits here as well, which is the composition
 * cnn_core will use.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_dense;

localparam WORD_BITS = 16;
localparam ACC_BITS  = 32;
localparam FRAC_BITS = 8;
localparam N_IN      = 8;
localparam N_CLASSES = 4;
localparam ADDR_BITS = 12;

integer errors;
integer i;

reg clk, rst_n;
reg start;
wire busy, done;

always #5 clk = ~clk;

task check;
    input signed [WORD_BITS-1:0] got, expected;
    input [127:0] name;
    begin
        if (got !== expected) begin
            $display("FAIL %0s: %0d, esperado %0d", name, got, expected);
            errors = errors + 1;
        end else
            $display("PASS %0s: %0d", name, got);
    end
endtask

/* activations memory */
reg  [ADDR_BITS-1:0] act_wr_addr;
reg  signed [WORD_BITS-1:0] act_wr_data;
reg  act_wr_en;
wire [ADDR_BITS-1:0] in_addr;
wire signed [WORD_BITS-1:0] in_data;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(N_IN), .ADDR_BITS(ADDR_BITS)) u_act_ram (
    .clk(clk), .wr_addr(act_wr_addr), .wr_data(act_wr_data), .wr_en(act_wr_en),
    .rd_addr(in_addr), .rd_data(in_data)
);

/* one weight memory per class, all at the same address */
wire [ADDR_BITS-1:0] w_addr;
wire signed [WORD_BITS-1:0] w0, w1, w2, w3;
wire signed [N_CLASSES*WORD_BITS-1:0] w_data = {w3, w2, w1, w0};

weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(N_IN), .ADDR_BITS(ADDR_BITS))
    u_w0 (.clk(clk), .addr(w_addr), .data(w0));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(N_IN), .ADDR_BITS(ADDR_BITS))
    u_w1 (.clk(clk), .addr(w_addr), .data(w1));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(N_IN), .ADDR_BITS(ADDR_BITS))
    u_w2 (.clk(clk), .addr(w_addr), .data(w2));
weight_rom #(.WORD_BITS(WORD_BITS), .DEPTH(N_IN), .ADDR_BITS(ADDR_BITS))
    u_w3 (.clk(clk), .addr(w_addr), .data(w3));

reg signed [N_CLASSES*WORD_BITS-1:0] bias_in;
wire signed [N_CLASSES*WORD_BITS-1:0] logits;
wire [1:0] class_idx;

dense #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .N_IN(N_IN), .N_CLASSES(N_CLASSES), .ADDR_BITS(ADDR_BITS)
) u_dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .in_addr(in_addr), .in_data(in_data),
    .w_addr(w_addr), .w_data(w_data), .bias_in(bias_in),
    .logits(logits)
);

argmax #(.WORD_BITS(WORD_BITS)) u_argmax (
    .logits(logits),
    .class_idx(class_idx)
);

task write_act;
    input [ADDR_BITS-1:0] addr;
    input signed [WORD_BITS-1:0] value;
    begin
        act_wr_addr = addr; act_wr_data = value; act_wr_en = 1;
        @(negedge clk);
        act_wr_en = 0;
    end
endtask

task run_layer;
    begin
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done); @(negedge clk);
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    start = 0; act_wr_en = 0; bias_in = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    /* Ativacoes Q8.8: 1.0, 2.0, 0.5, -1.0, 0.0, 3.0, -2.0, 1.5
     * Soma = 5.0 (1280 em Q8.8), usada pelas linhas uniformes abaixo. */
    write_act(0,  16'sd256);
    write_act(1,  16'sd512);
    write_act(2,  16'sd128);
    write_act(3, -16'sd256);
    write_act(4,  16'sd0);
    write_act(5,  16'sd768);
    write_act(6, -16'sd512);
    write_act(7,  16'sd384);

    /* Rodada 1
     *   classe 0: pesos 1.0     -> 5.0   (1280)
     *   classe 1: pesos 0.5     -> 2.5    (640)
     *   classe 2: pesos -1.0    -> -5.0 (-1280)
     *   classe 3: one-hot em 0  -> 1.0    (256)
     * vies: 0.25, -0.5, 1.0, 0.0 -> 64, -128, 256, 0
     * logits: 1344, 512, -1024, 256  -> argmax = 0 */
    for (i = 0; i < N_IN; i = i + 1) begin
        u_w0.mem[i] =  16'sd256;
        u_w1.mem[i] =  16'sd128;
        u_w2.mem[i] = -16'sd256;
        u_w3.mem[i] = (i == 0) ? 16'sd256 : 16'sd0;
    end
    bias_in = {16'sd0, 16'sd256, -16'sd128, 16'sd64};

    run_layer;

    check(logits[0*WORD_BITS +: WORD_BITS],  16'sd1344, "r1_classe0");
    check(logits[1*WORD_BITS +: WORD_BITS],  16'sd512,  "r1_classe1");
    check(logits[2*WORD_BITS +: WORD_BITS], -16'sd1024, "r1_classe2");
    check(logits[3*WORD_BITS +: WORD_BITS],  16'sd256,  "r1_onehot");
    check({14'd0, class_idx}, 16'sd0, "r1_argmax");

    /* Rodada 2, pesos trocados e vies zerado
     *   classe 0: 0            -> 0.0      (0)
     *   classe 1: 1.0          -> 5.0   (1280)
     *   classe 2: 2.0 so em 5  -> 6.0   (1536)
     *   classe 3: -0.5         -> -2.5  (-640)
     * logits: 0, 1280, 1536, -640 -> argmax = 2 */
    for (i = 0; i < N_IN; i = i + 1) begin
        u_w0.mem[i] =  16'sd0;
        u_w1.mem[i] =  16'sd256;
        u_w2.mem[i] = (i == 5) ? 16'sd512 : 16'sd0;
        u_w3.mem[i] = -16'sd128;
    end
    bias_in = 0;

    run_layer;

    check(logits[0*WORD_BITS +: WORD_BITS],  16'sd0,    "r2_zerado");
    check(logits[1*WORD_BITS +: WORD_BITS],  16'sd1280, "r2_classe1");
    check(logits[2*WORD_BITS +: WORD_BITS],  16'sd1536, "r2_onehot5");
    check(logits[3*WORD_BITS +: WORD_BITS], -16'sd640,  "r2_negativo");
    check({14'd0, class_idx}, 16'sd2, "r2_argmax");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
