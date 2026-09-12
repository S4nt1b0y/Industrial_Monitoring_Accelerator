/*
 * Testbench: cnn
 * The acceptance test. A real vibration window goes in through the same
 * block interface the UART frame buffer presents, and the RTL has to
 * come out with the right class.
 *
 * Stimulus is 05.Sim/cnn/vectors/samples_ch*.hex, written by
 * 03.Reference/tools/export_e2e_vectors_hex.py from one recording of
 * 0Nm_BPFI_03 (desgaste_rolamento), quantized to Q1.15 exactly as the
 * transmitter would. The expected class comes from the fixed-point model
 * that 03.Reference/tools/study_hw_quantization.py validated, so a
 * mismatch here means the RTL disagrees with the model rather than with
 * the label.
 *
 * Channels 2 and 3 are fed as zeros even though the network ignores
 * them. That is the real frame order, and it checks that the path keeps
 * `ready_o` asserted for blocks it discards -- if it did not, ingestion
 * would stall and the frame buffer would overflow.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_cnn;

localparam DATA_WIDTH = 16;
localparam N          = 64;
localparam N_SAMPLES  = 40960;
localparam ROUNDS     = N_SAMPLES / N;

localparam [1:0] CLASSE_ESPERADA = 2'd3;  // desgaste_rolamento

integer errors;
integer round, i;

reg clk, rst_n;
reg signed [N*DATA_WIDTH-1:0] sample_block_i;
reg [1:0] channel_i;
reg valid_i;

wire ready_o;
wire valid_o;
wire [1:0] class_o;

reg [DATA_WIDTH-1:0] ch0 [0:N_SAMPLES-1];
reg [DATA_WIDTH-1:0] ch1 [0:N_SAMPLES-1];

always #5 clk = ~clk;

cnn #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N),
    .KERNEL_FILE("../../04.RTL/cnn/weights/conv1_kernels.hex"),
    .CBIAS_FILE("../../04.RTL/cnn/weights/conv1_bias.hex"),
    .DBIAS_FILE("../../04.RTL/cnn/weights/dense_b.hex"),
    .DW0_FILE("../../04.RTL/cnn/weights/dense_w_c0.hex"),
    .DW1_FILE("../../04.RTL/cnn/weights/dense_w_c1.hex"),
    .DW2_FILE("../../04.RTL/cnn/weights/dense_w_c2.hex"),
    .DW3_FILE("../../04.RTL/cnn/weights/dense_w_c3.hex")
) u_dut (
    .clk(clk),
    .rst_n(rst_n),
    .sample_block_i(sample_block_i),
    .channel_i(channel_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .valid_o(valid_o),
    .class_o(class_o)
);

/* Hands one block over, holding valid until the path takes it. */
task offer_block;
    input [1:0] channel;
    begin
        channel_i = channel;
        valid_i   = 1;
        while (!ready_o) @(negedge clk);
        @(negedge clk);
        valid_i = 0;
    end
endtask

task load_block;
    input integer base;
    input integer which;
    begin
        for (i = 0; i < N; i = i + 1) begin
            if (which == 0)
                sample_block_i[i*DATA_WIDTH +: DATA_WIDTH] = ch0[base + i];
            else if (which == 1)
                sample_block_i[i*DATA_WIDTH +: DATA_WIDTH] = ch1[base + i];
            else
                sample_block_i[i*DATA_WIDTH +: DATA_WIDTH] = {DATA_WIDTH{1'b0}};
        end
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    valid_i = 0; channel_i = 0; sample_block_i = 0;

    $readmemh("vectors/samples_ch0.hex", ch0);
    $readmemh("vectors/samples_ch1.hex", ch1);

    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    round = 0;
    while (round < ROUNDS && !valid_o) begin
        load_block(round * N, 0); offer_block(2'd0);
        load_block(round * N, 1); offer_block(2'd1);
        load_block(round * N, 2); offer_block(2'd2);
        load_block(round * N, 3); offer_block(2'd3);
        round = round + 1;
        if (round % 64 == 0)
            $display("  %0d blocos por canal entregues", round);
    end

    /* valid_o may land while the last block is still being handed over. */
    if (!valid_o) begin
        i = 0;
        while (!valid_o && i < 2000000) begin
            @(negedge clk);
            i = i + 1;
        end
    end

    if (!valid_o) begin
        $display("FAIL sem classificacao apos %0d blocos por canal", round);
        errors = errors + 1;
    end else begin
        $display("classificou apos %0d blocos por canal", round);
        if (class_o !== CLASSE_ESPERADA) begin
            $display("FAIL classe_real: %0d, esperada %0d", class_o, CLASSE_ESPERADA);
            errors = errors + 1;
        end else
            $display("PASS classe_real: %0d (desgaste_rolamento)", class_o);
    end

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

initial begin
    #2000000000;
    $display("FAIL watchdog: simulacao travada");
    $finish;
end

endmodule
