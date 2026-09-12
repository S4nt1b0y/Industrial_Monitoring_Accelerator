/*
 * Testbench: prescale
 * Saturating left shift by 7. Checks the linear range, both saturation
 * edges and the boundary values either side of them, plus the measured
 * peak of the decimated stream (81) that set the shift in the first
 * place -- it must still be well inside the linear range.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_prescale;

localparam WORD_BITS = 16;
localparam SHIFT     = 7;

integer errors;

reg  signed [WORD_BITS-1:0] sample_in;
wire signed [WORD_BITS-1:0] sample_out;

prescale #(.WORD_BITS(WORD_BITS), .SHIFT(SHIFT)) u_dut (
    .sample_in(sample_in),
    .sample_out(sample_out)
);

task check;
    input signed [WORD_BITS-1:0] value, expected;
    input [127:0] name;
    begin
        sample_in = value;
        #1;
        if (sample_out !== expected) begin
            $display("FAIL %0s: %0d -> %0d, esperado %0d",
                     name, value, sample_out, expected);
            errors = errors + 1;
        end else
            $display("PASS %0s: %0d -> %0d", name, value, sample_out);
    end
endtask

initial begin
    errors = 0;

    check(16'sd0,      16'sd0,      "zero");
    check(16'sd1,      16'sd128,    "um");
    check(-16'sd1,    -16'sd128,    "menos_um");
    check(16'sd81,     16'sd10368,  "pico_medido");
    check(-16'sd81,   -16'sd10368,  "pico_negativo");

    // 255 << 7 = 32640, o maior que ainda cabe; 256 saturaria
    check(16'sd255,    16'sd32640,  "maior_linear");
    check(16'sd256,    16'sd32767,  "satura_pos");
    check(16'sd3000,   16'sd32767,  "satura_pos_alto");

    // -256 << 7 = -32768, exatamente o limite inferior
    check(-16'sd256,  -16'sd32768,  "menor_linear");
    check(-16'sd257,  -16'sd32768,  "satura_neg");
    check(-16'sd3000, -16'sd32768,  "satura_neg_alto");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
