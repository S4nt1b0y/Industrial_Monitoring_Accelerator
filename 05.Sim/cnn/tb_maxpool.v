/*
 * Testbench: maxpool
 * Two configurations:
 *   - 4x4 -> 2x2 with a hand-picked map that mixes positive and
 *     negative values, including one window that is entirely negative.
 *     That window is the point of the test: an unsigned comparator would
 *     report the most negative value as the maximum;
 *   - 32x32 -> 16x16, the real size one feature map runs at, to prove
 *     the addressing holds over 256 windows and not just a toy map.
 * Every expected maximum is written out in the comments.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_maxpool;

localparam WORD_BITS = 16;
localparam POOL      = 2;
localparam ADDR_BITS = 12;

integer errors;
integer y, x;

reg clk, rst_n;

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

/* ---------- DUT 1: 4x4 -> 2x2, com negativos ---------- */
localparam IMG1 = 4, OUT1 = IMG1 / POOL;

reg  start1;
wire busy1, done1;
wire [ADDR_BITS-1:0] in1_addr, out1_addr;
wire signed [WORD_BITS-1:0] in1_data, out1_word;
wire out1_we;

reg  [ADDR_BITS-1:0] in1_wr_addr, probe1_addr;
reg  signed [WORD_BITS-1:0] in1_wr_data;
reg  in1_wr_en;
wire signed [WORD_BITS-1:0] probe1_data;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG1*IMG1), .ADDR_BITS(ADDR_BITS)) u_in1_ram (
    .clk(clk), .wr_addr(in1_wr_addr), .wr_data(in1_wr_data), .wr_en(in1_wr_en),
    .rd_addr(in1_addr), .rd_data(in1_data)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(OUT1*OUT1), .ADDR_BITS(ADDR_BITS)) u_out1_ram (
    .clk(clk), .wr_addr(out1_addr), .wr_data(out1_word), .wr_en(out1_we),
    .rd_addr(probe1_addr), .rd_data(probe1_data)
);

maxpool #(
    .WORD_BITS(WORD_BITS), .POOL(POOL),
    .IMG_H(IMG1), .IMG_W(IMG1), .ADDR_BITS(ADDR_BITS)
) u_dut1 (
    .clk(clk), .rst_n(rst_n), .start(start1), .busy(busy1), .done(done1),
    .in_addr(in1_addr), .in_data(in1_data),
    .out_addr(out1_addr), .out_word(out1_word), .out_we(out1_we)
);

/* ---------- DUT 2: 32x32 -> 16x16, tamanho real ---------- */
localparam IMG2 = 32, OUT2 = IMG2 / POOL;

reg  start2;
wire busy2, done2;
wire [ADDR_BITS-1:0] in2_addr, out2_addr;
wire signed [WORD_BITS-1:0] in2_data, out2_word;
wire out2_we;

reg  [ADDR_BITS-1:0] in2_wr_addr, probe2_addr;
reg  signed [WORD_BITS-1:0] in2_wr_data;
reg  in2_wr_en;
wire signed [WORD_BITS-1:0] probe2_data;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(IMG2*IMG2), .ADDR_BITS(ADDR_BITS)) u_in2_ram (
    .clk(clk), .wr_addr(in2_wr_addr), .wr_data(in2_wr_data), .wr_en(in2_wr_en),
    .rd_addr(in2_addr), .rd_data(in2_data)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(OUT2*OUT2), .ADDR_BITS(ADDR_BITS)) u_out2_ram (
    .clk(clk), .wr_addr(out2_addr), .wr_data(out2_word), .wr_en(out2_we),
    .rd_addr(probe2_addr), .rd_data(probe2_data)
);

maxpool #(
    .WORD_BITS(WORD_BITS), .POOL(POOL),
    .IMG_H(IMG2), .IMG_W(IMG2), .ADDR_BITS(ADDR_BITS)
) u_dut2 (
    .clk(clk), .rst_n(rst_n), .start(start2), .busy(busy2), .done(done2),
    .in_addr(in2_addr), .in_data(in2_data),
    .out_addr(out2_addr), .out_word(out2_word), .out_we(out2_we)
);

task write_in1;
    input [ADDR_BITS-1:0] addr;
    input signed [WORD_BITS-1:0] value;
    begin
        in1_wr_addr = addr; in1_wr_data = value; in1_wr_en = 1;
        @(negedge clk);
        in1_wr_en = 0;
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    start1 = 0; start2 = 0;
    in1_wr_en = 0; in2_wr_en = 0;
    probe1_addr = 0; probe2_addr = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    /* Mapa 4x4 (linha por linha):
     *    1    -5     7     3
     *   -2    -9     0     6
     *   -4    -8   -20   -11
     *   -3    -7   -15    -6
     * Janelas 2x2:
     *   (0,0): {1,-5,-2,-9}      -> 1
     *   (0,1): {7,3,0,6}         -> 7
     *   (1,0): {-4,-8,-3,-7}     -> -3   (janela toda negativa)
     *   (1,1): {-20,-11,-15,-6}  -> -6   (janela toda negativa)
     */
    write_in1(0,  16'sd1);   write_in1(1,  -16'sd5);  write_in1(2,  16'sd7);   write_in1(3,  16'sd3);
    write_in1(4,  -16'sd2);  write_in1(5,  -16'sd9);  write_in1(6,  16'sd0);   write_in1(7,  16'sd6);
    write_in1(8,  -16'sd4);  write_in1(9,  -16'sd8);  write_in1(10, -16'sd20); write_in1(11, -16'sd11);
    write_in1(12, -16'sd3);  write_in1(13, -16'sd7);  write_in1(14, -16'sd15); write_in1(15, -16'sd6);

    @(negedge clk); start1 = 1;
    @(negedge clk); start1 = 0;
    wait (done1); @(negedge clk);

    probe1_addr = 0; #1; check(probe1_data,  16'sd1,   "4x4_positivo");
    probe1_addr = 1; #1; check(probe1_data,  16'sd7,   "4x4_maior");
    probe1_addr = 2; #1; check(probe1_data, -16'sd3,   "4x4_negativo");
    probe1_addr = 3; #1; check(probe1_data, -16'sd6,   "4x4_negativo2");

    /* Mapa 32x32: valor = y*32 + x - 512, entao a metade de cima e
     * negativa e a de baixo positiva. O maximo de cada janela e sempre o
     * canto inferior direito, (2y+1)*32 + (2x+1) - 512. */
    for (y = 0; y < IMG2; y = y + 1)
        for (x = 0; x < IMG2; x = x + 1) begin
            in2_wr_addr = y*IMG2 + x;
            in2_wr_data = y*IMG2 + x - 512;
            in2_wr_en = 1;
            @(negedge clk);
            in2_wr_en = 0;
        end

    @(negedge clk); start2 = 1;
    @(negedge clk); start2 = 0;
    wait (done2); @(negedge clk);

    // (0,0): max de {-512,-511,-480,-479} -> -479
    probe2_addr = 0;              #1; check(probe2_data, -16'sd479, "32x32_primeiro");
    // (7,7): (15*32+15)-512 = -17
    probe2_addr = 7*OUT2 + 7;     #1; check(probe2_data, -16'sd17,  "32x32_meio");
    // (8,0): (17*32+1)-512 = 33
    probe2_addr = 8*OUT2 + 0;     #1; check(probe2_data,  16'sd33,  "32x32_positivo");
    // (15,15): (31*32+31)-512 = 511, prova que fecha ate o fim do mapa
    probe2_addr = 15*OUT2 + 15;   #1; check(probe2_data,  16'sd511, "32x32_fim");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
