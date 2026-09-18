/*
 * Testbench: block_feeder
 * Feeds two blocks with different channel tags and checks that every
 * sample comes out once, in order, with its tag. Sample i of block 0 is
 * i-100, so the stream crosses zero.
 *
 * The second block is fed with dst_busy toggling: no sample may be
 * emitted while busy, and none may be lost or repeated.
 * A watchdog ends the run if the handshake deadlocks.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_block_feeder;

localparam WORD_BITS = 16;
localparam N         = 64;

integer errors;
integer i;
integer received;

reg clk, rst_n;
reg signed [N*WORD_BITS-1:0] sample_block_i;
reg [1:0] channel_i;
reg valid_i;
reg dst_busy;

wire ready_o;
wire sample_valid_o;
wire signed [WORD_BITS-1:0] sample_o;
wire [1:0] channel_o;

reg signed [WORD_BITS-1:0] expected_sample;
reg [1:0] expected_channel;
reg checking;

always #5 clk = ~clk;

block_feeder #(.WORD_BITS(WORD_BITS), .N(N)) u_dut (
    .clk(clk), .rst_n(rst_n),
    .sample_block_i(sample_block_i),
    .channel_i(channel_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .dst_busy(dst_busy),
    .sample_valid_o(sample_valid_o),
    .sample_o(sample_o),
    .channel_o(channel_o)
);

/* Checks each sample as it is emitted, so a repeated or skipped sample
 * fails at once. */
always @(posedge clk) begin
    if (rst_n && checking && sample_valid_o) begin
        if (dst_busy) begin
            $display("FAIL emitiu com dst_busy alto no indice %0d", received);
            errors = errors + 1;
        end
        if (sample_o !== expected_sample) begin
            $display("FAIL amostra %0d: %0d, esperado %0d",
                     received, sample_o, expected_sample);
            errors = errors + 1;
        end
        if (channel_o !== expected_channel) begin
            $display("FAIL canal na amostra %0d: %0d, esperado %0d",
                     received, channel_o, expected_channel);
            errors = errors + 1;
        end
        received        <= received + 1;
        expected_sample <= expected_sample + 16'sd1;
    end
end

/* Offers the block once: holds valid until a negedge with ready high
 * (the block is loaded at the next posedge), then drops it. */
task offer_block;
    begin
        valid_i = 1;
        while (!ready_o) @(negedge clk);
        @(negedge clk);
        valid_i = 0;
    end
endtask

task fill_block;
    input [1:0] channel;
    input signed [WORD_BITS-1:0] first;
    begin
        for (i = 0; i < N; i = i + 1)
            sample_block_i[i*WORD_BITS +: WORD_BITS] = first + i;
        channel_i        = channel;
        expected_sample  = first;
        expected_channel = channel;
        received         = 0;
    end
endtask

task expect_count;
    input [127:0] name;
    begin
        if (received == N)
            $display("PASS %0s: %0d amostras", name, received);
        else begin
            $display("FAIL %0s: %0d de %0d", name, received, N);
            errors = errors + 1;
        end
    end
endtask

initial begin
    clk = 0; rst_n = 0; errors = 0;
    valid_i = 0; dst_busy = 0; channel_i = 0;
    sample_block_i = 0; received = 0; checking = 0;
    expected_sample = 0; expected_channel = 0;
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    checking = 1;

    /* Block 0, channel 0, no stall: samples -100..-37 */
    fill_block(2'd0, -16'sd100);
    offer_block;
    wait (received == N);
    @(negedge clk);
    expect_count("bloco0_completo");

    /* Block 1, channel 2, with dst_busy toggling */
    fill_block(2'd2, 16'sd1000);
    fork
        offer_block;
        begin : stall_gen
            while (received < N) begin
                dst_busy = 1;
                repeat (3) @(negedge clk);
                dst_busy = 0;
                repeat (2) @(negedge clk);
            end
            dst_busy = 0;
        end
    join

    wait (received == N);
    @(negedge clk);
    expect_count("bloco1_com_stall");

    /* After the last block the feeder accepts again */
    @(negedge clk);
    if (ready_o)
        $display("PASS pronto_de_novo");
    else begin
        $display("FAIL pronto_de_novo: ready_o baixo apos terminar");
        errors = errors + 1;
    end

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

/* Watchdog limit, well above the longest expected run: 2 blocks
 * with a stall in 3 of every 5 cycles. */
initial begin
    #200000;
    $display("FAIL watchdog: simulacao travada");
    $finish;
end

endmodule
