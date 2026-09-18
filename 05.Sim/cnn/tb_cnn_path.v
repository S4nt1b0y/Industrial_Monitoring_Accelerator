/*
 * Testbench: cnn_path
 * Same acceptance case as tb_cnn, through the stream interface: samples
 * arrive one per cycle, tagged with their channel. No block_feeder.
 *
 * Channels 0 and 1 are interleaved sample by sample, so a sample routed
 * to the wrong decimator changes the class.
 * Channel 2 is fed with zeros to check that unused tags are ignored.
 * Stimulus and expected class are the same as in tb_cnn.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_cnn_path;

localparam WORD_BITS = 16;
localparam N_SAMPLES = 40960;

/* Default: 0Nm_BPFI_03, bearing wear (class 3). The Makefile overrides
 * both for 0Nm_Normal (vectors/normal/, class 0). */
parameter       VEC_DIR         = "vectors/";
parameter [1:0] CLASSE_ESPERADA = 2'd3;

integer errors;
integer i;

reg clk, rst_n;
reg sample_valid;
reg signed [WORD_BITS-1:0] sample_in;
reg [1:0] sample_channel;

wire stall;
wire valid_o;
wire [1:0] class_o;

reg [WORD_BITS-1:0] ch0 [0:N_SAMPLES-1];
reg [WORD_BITS-1:0] ch1 [0:N_SAMPLES-1];

always #5 clk = ~clk;

cnn_path #(
    .WORD_BITS(WORD_BITS),
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
    .sample_valid(sample_valid),
    .sample_in(sample_in),
    .sample_channel(sample_channel),
    .stall(stall),
    .valid_o(valid_o),
    .class_o(class_o)
);

/* Sends one sample after stall goes low. */
task send;
    input [1:0] channel;
    input [WORD_BITS-1:0] value;
    begin
        while (stall) @(negedge clk);
        sample_channel = channel;
        sample_in      = value;
        sample_valid   = 1;
        @(negedge clk);
        sample_valid   = 0;
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    sample_valid = 0; sample_in = 0; sample_channel = 0;

    $readmemh({VEC_DIR, "samples_ch0.hex"}, ch0);
    $readmemh({VEC_DIR, "samples_ch1.hex"}, ch1);

    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    i = 0;
    while (i < N_SAMPLES && !valid_o) begin
        send(2'd0, ch0[i]);
        send(2'd1, ch1[i]);
        send(2'd2, 16'd0);   // tag not used by the network
        i = i + 1;
        if (i % 4096 == 0)
            $display("  %0d amostras por canal entregues", i);
    end

    if (!valid_o) begin
        i = 0;
        while (!valid_o && i < 2000000) begin
            @(negedge clk);
            i = i + 1;
        end
    end

    if (!valid_o) begin
        $display("FAIL sem classificacao");
        errors = errors + 1;
    end else begin
        if (class_o !== CLASSE_ESPERADA) begin
            $display("FAIL classe_stream: %0d, esperada %0d", class_o, CLASSE_ESPERADA);
            errors = errors + 1;
        end else
            $display("PASS classe_stream: %0d", class_o);
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
