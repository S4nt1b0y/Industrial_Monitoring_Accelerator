/*
 * Testbench: cnn
 * Acceptance test: a real vibration window goes in through the block
 * interface and the class must match the expected one.
 *
 * Stimulus: vectors/samples_ch0.hex and samples_ch1.hex, one recording
 * of 0Nm_BPFI_03 (bearing wear) quantized to Q1.15 as the transmitter
 * sends it. vectors/normal/ holds 0Nm_Normal (normal operation), run by
 * `make tb_cnn_normal`.
 * The expected class is the output of the fixed-point reference model
 * for the same window.
 * Channels 2 and 3 are fed as zeros, as in the real frame order; this
 * checks that ready_o stays high for discarded blocks.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_cnn;

localparam DATA_WIDTH = 16;
localparam N          = 64;
localparam N_SAMPLES  = 40960;
localparam ROUNDS     = N_SAMPLES / N;

/* Default: 0Nm_BPFI_03, bearing wear (class 3). The Makefile overrides
 * both for 0Nm_Normal (vectors/normal/, class 0). */
parameter       VEC_DIR         = "vectors/";
parameter [1:0] CLASSE_ESPERADA = 2'd3;

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

/* Offers one block, holding valid until it is accepted. */
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

    $readmemh({VEC_DIR, "samples_ch0.hex"}, ch0);
    $readmemh({VEC_DIR, "samples_ch1.hex"}, ch1);

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

    /* valid_o can rise while the last block is still being offered. */
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
            $display("PASS classe_real: %0d", class_o);
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
