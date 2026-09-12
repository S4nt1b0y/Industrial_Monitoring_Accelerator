/*
 * Testbench: argmax
 * Directed cases over the 4 logits: a winner in each position, all
 * logits negative (an unsigned compare would pick the wrong one), a tie
 * between two classes and a four-way tie. The tie cases are the point:
 * the reference model uses numpy's argmax, which keeps the lowest
 * index, and a naive >= comparison would keep the highest instead.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_argmax;

localparam WORD_BITS = 16;

integer errors;

reg signed [4*WORD_BITS-1:0] logits;
wire [1:0] class_idx;

argmax #(.WORD_BITS(WORD_BITS)) u_dut (
    .logits(logits),
    .class_idx(class_idx)
);

task drive;
    input signed [WORD_BITS-1:0] l0, l1, l2, l3;
    input [1:0] expected;
    input [127:0] name;
    begin
        logits[0*WORD_BITS +: WORD_BITS] = l0;
        logits[1*WORD_BITS +: WORD_BITS] = l1;
        logits[2*WORD_BITS +: WORD_BITS] = l2;
        logits[3*WORD_BITS +: WORD_BITS] = l3;
        #1;
        if (class_idx !== expected) begin
            $display("FAIL %0s: %0d, esperado %0d", name, class_idx, expected);
            errors = errors + 1;
        end else
            $display("PASS %0s: %0d", name, class_idx);
    end
endtask

initial begin
    errors = 0;

    drive( 16'sd100,  16'sd10,   16'sd20,   16'sd30,  2'd0, "vence_0");
    drive( 16'sd10,   16'sd100,  16'sd20,   16'sd30,  2'd1, "vence_1");
    drive( 16'sd10,   16'sd20,   16'sd100,  16'sd30,  2'd2, "vence_2");
    drive( 16'sd10,   16'sd20,   16'sd30,   16'sd100, 2'd3, "vence_3");

    // todos negativos: -5 e o maior; comparacao sem sinal escolheria -100
    drive(-16'sd50,  -16'sd100, -16'sd5,   -16'sd80,  2'd2, "todos_negativos");
    drive(-16'sd5,   -16'sd50,  -16'sd100, -16'sd80,  2'd0, "neg_vence_0");

    // empates: sempre o menor indice, como o numpy argmax
    drive( 16'sd70,   16'sd70,   16'sd10,   16'sd20,  2'd0, "empate_0_1");
    drive( 16'sd10,   16'sd20,   16'sd70,   16'sd70,  2'd2, "empate_2_3");
    drive( 16'sd10,   16'sd70,   16'sd70,   16'sd20,  2'd1, "empate_1_2");
    drive( 16'sd70,   16'sd70,   16'sd70,   16'sd70,  2'd0, "empate_total");

    // margem de 1 LSB, nos dois sentidos
    drive( 16'sd70,   16'sd71,   16'sd10,   16'sd20,  2'd1, "margem_1lsb");
    drive(-16'sd1,     16'sd0,   -16'sd2,   -16'sd3,  2'd1, "cruza_zero");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
