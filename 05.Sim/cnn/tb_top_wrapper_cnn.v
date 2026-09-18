/*
 * Testbench: top_wrapper, CNN path
 * Acceptance case through the board top: SW[0] = 1, with the frame
 * buffer, the path selection in top_wrapper and the LED decoder in the
 * loop. The class is checked on LEDR.
 *
 * Bytes are forced onto uart_data and uart_valid at the uart_rx output.
 * Serial transmission of the 640 frames would take about 340 million
 * cycles; uart_rx is covered by 05.Sim/top/tb_top_wrapper.v.
 * Frame format, as sent by uart_tx_cnn.py: 4 channels x 64 samples,
 * int16 big-endian, order x_A, y_A, x_B, y_B. Channels 2 and 3 are sent
 * as zeros. Each channel block starts after the frame buffer has handed
 * over the previous one.
 * Checks: one-hot class on LEDR, decoded as in top_wrapper (0 normal,
 * 1 misalignment, 2 unbalance, 3 bearing wear); no frame buffer
 * overflow; no block offered to the ML path while SW[0] = 1.
 * VEC_DIR and CLASSE_ESPERADA select the recording, as in tb_cnn.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_top_wrapper_cnn;

parameter       VEC_DIR         = "vectors/";
parameter [1:0] CLASSE_ESPERADA = 2'd3;

localparam DATA_WIDTH = 16;
localparam N          = 64;
localparam N_SAMPLES  = 40960;
localparam ROUNDS     = N_SAMPLES / N;

reg clk, rst_n;
reg       inj_valid;
reg [7:0] inj_data;

wire [3:0] KEY = {3'b111, rst_n};
wire [9:0] SW  = 10'b00_0000_0001;
wire [9:0] LEDR;

reg [DATA_WIDTH-1:0] ch0 [0:N_SAMPLES-1];
reg [DATA_WIDTH-1:0] ch1 [0:N_SAMPLES-1];

integer errors;
integer round, ch, i;

reg       seen_valid;
reg [1:0] seen_class;
reg       seen_overflow;
reg       seen_ml_offer;

always #5 clk = ~clk;

top_wrapper #(
    .CNN_WEIGHTS_DIR("../../04.RTL/cnn/weights/")
) dut (
    .CLOCK_50(clk),
    .KEY(KEY),
    .SW(SW),
    .GPIO_0_0(1'b1),
    .LEDR(LEDR)
);

always @(posedge clk) begin
    if (!rst_n) begin
        seen_valid    <= 1'b0;
        seen_class    <= 2'd0;
        seen_overflow <= 1'b0;
        seen_ml_offer <= 1'b0;
    end else begin
        if (dut.cnn_valid_o && !seen_valid) begin
            seen_valid <= 1'b1;
            seen_class <= dut.cnn_class_o;
        end
        if (dut.frame_overflow)
            seen_overflow <= 1'b1;
        if (dut.ml_valid_i)
            seen_ml_offer <= 1'b1;
    end
end

task send_byte;
    input [7:0] value;
    begin
        inj_data  = value;
        inj_valid = 1'b1;
        @(negedge clk);
        inj_valid = 1'b0;
        @(negedge clk);
    end
endtask

/* Sends one 64-sample channel block after the frame buffer is empty. */
task send_block;
    input integer base;
    input integer which;
    reg [DATA_WIDTH-1:0] word;
    begin
        while (dut.frame_valid) @(negedge clk);
        for (i = 0; i < N; i = i + 1) begin
            if (which == 0)
                word = ch0[base + i];
            else if (which == 1)
                word = ch1[base + i];
            else
                word = {DATA_WIDTH{1'b0}};
            send_byte(word[15:8]);
            send_byte(word[7:0]);
        end
    end
endtask

function [3:0] led_of;
    input [1:0] cls;
    begin
        case (cls)
            2'd0: led_of = 4'b0001;
            2'd1: led_of = 4'b0100;
            2'd2: led_of = 4'b0010;
            default: led_of = 4'b1000;
        endcase
    end
endfunction

initial begin
    clk = 0; rst_n = 0; errors = 0;
    inj_valid = 0; inj_data = 0;

    force dut.uart_valid = inj_valid;
    force dut.uart_data  = inj_data;

    $readmemh({VEC_DIR, "samples_ch0.hex"}, ch0);
    $readmemh({VEC_DIR, "samples_ch1.hex"}, ch1);

    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

    round = 0;
    while (round < ROUNDS && !seen_valid) begin
        for (ch = 0; ch < 4; ch = ch + 1)
            send_block(round * N, ch);
        round = round + 1;
        if (round % 64 == 0)
            $display("  %0d quadros entregues", round);
    end

    i = 0;
    while (!seen_valid && i < 2000000) begin
        @(negedge clk);
        i = i + 1;
    end
    repeat (4) @(negedge clk);

    if (!seen_valid) begin
        $display("FAIL sem classificacao apos %0d quadros", round);
        errors = errors + 1;
    end else begin
        $display("classificou apos %0d quadros", round);
        if (seen_class !== CLASSE_ESPERADA) begin
            $display("FAIL classe_topo: %0d, esperada %0d", seen_class, CLASSE_ESPERADA);
            errors = errors + 1;
        end else
            $display("PASS classe_topo: %0d", seen_class);

        if (LEDR[3:0] !== led_of(CLASSE_ESPERADA)) begin
            $display("FAIL ledr: %b, esperado %b", LEDR[3:0], led_of(CLASSE_ESPERADA));
            errors = errors + 1;
        end else
            $display("PASS ledr: %b", LEDR[3:0]);
    end

    if (seen_overflow) begin
        $display("FAIL overflow no buffer de quadros");
        errors = errors + 1;
    end else
        $display("PASS sem_overflow");

    if (seen_ml_offer) begin
        $display("FAIL caminho ML recebeu bloco com SW[0] = 1");
        errors = errors + 1;
    end else
        $display("PASS rota_isolada");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

initial begin
    #(64'd20_000_000_000);
    $display("FAIL watchdog: simulacao travada");
    $finish;
end

endmodule
