/*
 * Testbench: cnn_core
 * Two runs at the real 32x32x2 size, both with weights chosen so the
 * expected logits can be worked out by hand.
 *
 * Run 1 targets the flatten offset, which is the one thing in this
 * design that would corrupt results silently. All kernels are zero and
 * each filter gets a distinct convolution bias, so after ReLU and
 * pooling every element of filter f's block holds f+1. Each class then
 * has a single dense weight of 1.0 placed in a different filter's block,
 * so the four logits read back the four biases. If the filter stride
 * were missing from the flatten address, all four classes would read
 * filter 0 and the logits would tie instead.
 *
 * Run 2 targets the data path: one kernel tap of 1.0 on channel 0 turns
 * the convolution into a copy, so a constant spectrogram of 1000 comes
 * out of normalize at (1000*287>>8)-116 = 1005 and reaches the logit
 * unchanged.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_cnn_core;

localparam WORD_BITS  = 16;
localparam ACC_BITS   = 32;
localparam FRAC_BITS  = 8;
localparam IMG        = 32;
localparam KSIZE      = 3;
localparam N_CHANNELS = 2;
localparam N_FILTERS  = 8;
localparam POOL       = 2;
localparam N_CLASSES  = 4;
localparam ADDR_BITS  = 12;
localparam GAIN       = 287;
localparam OFFSET     = 116;

localparam IMG_ELEMS  = IMG * IMG;                  // 1024
localparam POOL_ELEMS = (IMG/POOL) * (IMG/POOL);    // 256
localparam FLAT_ELEMS = N_FILTERS * POOL_ELEMS;     // 2048
localparam K_WORDS    = KSIZE * KSIZE * N_CHANNELS; // 18

integer errors;
integer i, f;

reg clk, rst_n;
reg start;
wire busy, done;
wire [1:0] class_o;

always #5 clk = ~clk;

task check;
    input [1:0] got, expected;
    input [127:0] name;
    begin
        if (got !== expected) begin
            $display("FAIL %0s: classe %0d, esperada %0d", name, got, expected);
            errors = errors + 1;
        end else
            $display("PASS %0s: classe %0d", name, got);
    end
endtask

/* spectrogram memories, filled by the testbench the way the two
 * spectrogram blocks would fill them in the real design */
reg  [ADDR_BITS-1:0] spec_wr_addr;
reg  signed [WORD_BITS-1:0] spec_wr_data;
reg  spec_wr_en0, spec_wr_en1;

wire [ADDR_BITS-1:0] spec_addr;
wire signed [WORD_BITS-1:0] spec_data_ch0, spec_data_ch1;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_ELEMS), .ADDR_BITS(ADDR_BITS)) u_spec0 (
    .clk(clk), .wr_addr(spec_wr_addr), .wr_data(spec_wr_data), .wr_en(spec_wr_en0),
    .rd_addr(spec_addr), .rd_data(spec_data_ch0)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_ELEMS), .ADDR_BITS(ADDR_BITS)) u_spec1 (
    .clk(clk), .wr_addr(spec_wr_addr), .wr_data(spec_wr_data), .wr_en(spec_wr_en1),
    .rd_addr(spec_addr), .rd_data(spec_data_ch1)
);

cnn_core #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .IMG_H(IMG), .IMG_W(IMG), .KSIZE(KSIZE),
    .N_CHANNELS(N_CHANNELS), .N_FILTERS(N_FILTERS), .POOL(POOL),
    .N_CLASSES(N_CLASSES), .ADDR_BITS(ADDR_BITS),
    .GAIN(GAIN), .OFFSET(OFFSET)
) u_dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .busy(busy), .done(done),
    .spec_addr(spec_addr),
    .spec_data_ch0(spec_data_ch0),
    .spec_data_ch1(spec_data_ch1),
    .class_o(class_o)
);

task fill_spectrogram;
    input signed [WORD_BITS-1:0] value;
    begin
        for (i = 0; i < IMG_ELEMS; i = i + 1) begin
            spec_wr_addr = i; spec_wr_data = value;
            spec_wr_en0 = 1; spec_wr_en1 = 1;
            @(negedge clk);
        end
        spec_wr_en0 = 0; spec_wr_en1 = 0;
    end
endtask

task zero_weights;
    begin
        for (i = 0; i < N_FILTERS*K_WORDS; i = i + 1)
            u_dut.u_kernel_rom.mem[i] = 16'sd0;
        for (i = 0; i < N_FILTERS; i = i + 1)
            u_dut.u_conv_bias_rom.mem[i] = 16'sd0;
        for (i = 0; i < FLAT_ELEMS; i = i + 1) begin
            u_dut.u_dw0.mem[i] = 16'sd0;
            u_dut.u_dw1.mem[i] = 16'sd0;
            u_dut.u_dw2.mem[i] = 16'sd0;
            u_dut.u_dw3.mem[i] = 16'sd0;
        end
        for (i = 0; i < N_CLASSES; i = i + 1)
            u_dut.u_dense_bias_rom.mem[i] = 16'sd0;
    end
endtask

task run_core;
    begin
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait (done); @(negedge clk);
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    start = 0; spec_wr_en0 = 0; spec_wr_en1 = 0;
    spec_wr_addr = 0; spec_wr_data = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    /* ---- Run 1: offset do flatten ----
     * kernels zerados, vies do filtro f = (f+1).0 em Q8.8.
     * Apos ReLU e pooling, todo o bloco do filtro f vale (f+1).0.
     * Peso 1.0 da classe c num indice dentro do bloco de um filtro
     * diferente por classe:
     *   classe 0 -> filtro 0 -> logit 1.0  (256)
     *   classe 1 -> filtro 3 -> logit 4.0  (1024)
     *   classe 2 -> filtro 7 -> logit 8.0  (2048)  <- vence
     *   classe 3 -> filtro 1 -> logit 2.0  (512)
     */
    fill_spectrogram(16'sd0);
    zero_weights;
    for (f = 0; f < N_FILTERS; f = f + 1)
        u_dut.u_conv_bias_rom.mem[f] = (f + 1) * 256;
    u_dut.u_dw0.mem[0*POOL_ELEMS + 5] = 16'sd256;
    u_dut.u_dw1.mem[3*POOL_ELEMS + 7] = 16'sd256;
    u_dut.u_dw2.mem[7*POOL_ELEMS + 0] = 16'sd256;
    u_dut.u_dw3.mem[1*POOL_ELEMS + 1] = 16'sd256;

    run_core;
    check(class_o, 2'd2, "offset_flatten");
    check({1'b0, u_dut.logits[2*WORD_BITS +: WORD_BITS] == 16'sd2048}, 2'd1,
          "logit_filtro7");
    check({1'b0, u_dut.logits[1*WORD_BITS +: WORD_BITS] == 16'sd1024}, 2'd1,
          "logit_filtro3");
    check({1'b0, u_dut.logits[3*WORD_BITS +: WORD_BITS] == 16'sd512}, 2'd1,
          "logit_filtro1");

    /* ---- Run 2: caminho de dados ----
     * espectrograma constante 1000 -> normalize -> (1000*287>>8)-116 = 1005.
     * Um unico tap 1.0 no centro do kernel do filtro 0, canal 0, faz a
     * convolucao copiar a entrada. Peso 1.0 da classe 1 no indice 0 do
     * bloco do filtro 0 -> logit 1005; as outras classes ficam em 0.
     * Centro do kernel: ky=1, kx=1, c=0 -> (1*3+1)*2+0 = 8.
     */
    fill_spectrogram(16'sd1000);
    zero_weights;
    u_dut.u_kernel_rom.mem[0*K_WORDS + 8] = 16'sd256;
    u_dut.u_dw1.mem[0] = 16'sd256;

    run_core;
    check(class_o, 2'd1, "caminho_dados");
    check({1'b0, u_dut.logits[1*WORD_BITS +: WORD_BITS] == 16'sd1005}, 2'd1,
          "logit_norm");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

/* Uma passada completa leva ~160 mil ciclos; folga larga sobre isso. */
initial begin
    #40000000;
    $display("FAIL watchdog: simulacao travada");
    $finish;
end

endmodule
