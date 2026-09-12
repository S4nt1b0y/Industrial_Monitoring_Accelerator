/*
 * Testbench: normalize
 * Streams 8 magnitudes through the z-score bridge with the real
 * constants (GAIN=287, OFFSET=116) and checks each Q8.8 result, worked
 * out by hand as (mag*287 >> 8) - 116.
 *
 * Two of the values are anchors rather than arbitrary picks: magnitude
 * 3370 is the peak the hardware spectrogram actually reached in
 * 03.Reference/tools/study_hw_quantization.py, and it must come out at
 * 3662, the same number that study reported as the CNN's largest input.
 * Magnitude 32767 checks the saturation path, which the measured data
 * never reaches.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_normalize;

localparam WORD_BITS = 16;
localparam ACC_BITS  = 32;
localparam FRAC_BITS = 8;
localparam N_ELEMS   = 8;
localparam ADDR_BITS = 12;
localparam GAIN      = 287;
localparam OFFSET    = 116;

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

reg  [ADDR_BITS-1:0] in_wr_addr, probe_addr;
reg  signed [WORD_BITS-1:0] in_wr_data;
reg  in_wr_en;

wire [ADDR_BITS-1:0] in_addr, out_addr;
wire signed [WORD_BITS-1:0] in_data, out_word, probe_data;
wire out_we;

ram #(.WORD_BITS(WORD_BITS), .DEPTH(N_ELEMS), .ADDR_BITS(ADDR_BITS)) u_in_ram (
    .clk(clk), .wr_addr(in_wr_addr), .wr_data(in_wr_data), .wr_en(in_wr_en),
    .rd_addr(in_addr), .rd_data(in_data)
);

ram #(.WORD_BITS(WORD_BITS), .DEPTH(N_ELEMS), .ADDR_BITS(ADDR_BITS)) u_out_ram (
    .clk(clk), .wr_addr(out_addr), .wr_data(out_word), .wr_en(out_we),
    .rd_addr(probe_addr), .rd_data(probe_data)
);

normalize #(
    .WORD_BITS(WORD_BITS), .ACC_BITS(ACC_BITS), .FRAC_BITS(FRAC_BITS),
    .N_ELEMS(N_ELEMS), .ADDR_BITS(ADDR_BITS), .GAIN(GAIN), .OFFSET(OFFSET)
) u_dut (
    .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
    .in_addr(in_addr), .in_data(in_data),
    .out_addr(out_addr), .out_word(out_word), .out_we(out_we)
);

task write_in;
    input [ADDR_BITS-1:0] addr;
    input [WORD_BITS-1:0] value;
    begin
        in_wr_addr = addr; in_wr_data = value; in_wr_en = 1;
        @(negedge clk);
        in_wr_en = 0;
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    start = 0; in_wr_en = 0; probe_addr = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    write_in(0, 16'd0);
    write_in(1, 16'd1);
    write_in(2, 16'd100);
    write_in(3, 16'd200);
    write_in(4, 16'd1000);
    write_in(5, 16'd3370);
    write_in(6, 16'd5000);
    write_in(7, 16'd32767);

    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait (done); @(negedge clk);

    // 0*287>>8 = 0        -> 0   - 116 = -116  (o piso do espectrograma)
    probe_addr = 0; #1; check(probe_data, -16'sd116,  "zero");
    // 1*287>>8 = 1        -> 1   - 116 = -115
    probe_addr = 1; #1; check(probe_data, -16'sd115,  "um");
    // 100*287>>8 = 112    -> 112 - 116 = -4
    probe_addr = 2; #1; check(probe_data, -16'sd4,    "cruza_zero");
    // 200*287>>8 = 224    -> 224 - 116 = 108
    probe_addr = 3; #1; check(probe_data,  16'sd108,  "positivo");
    // 1000*287>>8 = 1121  -> 1121 - 116 = 1005
    probe_addr = 4; #1; check(probe_data,  16'sd1005, "mil");
    // 3370*287>>8 = 3778  -> 3778 - 116 = 3662, o pico medido no estudo
    probe_addr = 5; #1; check(probe_data,  16'sd3662, "pico_do_estudo");
    // 5000*287>>8 = 5605  -> 5605 - 116 = 5489
    probe_addr = 6; #1; check(probe_data,  16'sd5489, "cinco_mil");
    // 32767*287>>8 = 36734, satura em 32767 -> 32767 - 116 = 32651
    probe_addr = 7; #1; check(probe_data,  16'sd32651, "satura");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
