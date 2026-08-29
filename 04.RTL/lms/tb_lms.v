//======================================================
// ARQUIVO: tb_lms.v
//
// TESTE 2 - LMS COM AMOSTRAS VARIADAS
//
// Objetivos:
//   1. Testar números positivos e negativos
//   2. Verificar o histórico x_mem[0..7]
//   3. Verificar os 8 taps
//   4. Verificar atualização dos pesos
//   5. Verificar busy/valid
//   6. Observar y(n) e e(n)
//
// Formato: Q1.15
//
// Exemplos:
//   +0,5   ->  16384
//   -0,5   -> -16384
//   +0,25  ->   8192
//   -0,25  ->  -8192
//======================================================

`timescale 1ns/1ps

module tb_lms;

    localparam DATA_W = 16;
    localparam TAPS   = 8;

    //==================================================
    // CLOCK
    //==================================================

    reg clk;

    initial begin
        clk = 1'b0;
        forever #10 clk = ~clk; // 50 MHz
    end

    //==================================================
    // CONTROLE
    //==================================================

    reg reset;
    reg start;

    //==================================================
    // ENTRADAS
    //==================================================

    reg signed [DATA_W-1:0] x_in;
    reg signed [DATA_W-1:0] d_in;

    //==================================================
    // SAÍDAS
    //==================================================

    wire signed [DATA_W-1:0] y_out;
    wire signed [DATA_W-1:0] e_out;

    wire valid;
    wire busy;

    //==================================================
    // INSTÂNCIA DO LMS
    //==================================================

    lms_top #(
        .DATA_W(DATA_W),
        .TAPS(TAPS),
        .FRAC_W(15),
        .MU_SHIFT(6)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),

        .x_in(x_in),
        .d_in(d_in),

        .e_out(e_out),
        .y_out(y_out),

        .valid(valid),
        .busy(busy)
    );

    //==================================================
    // CONVERSÃO Q1.15 -> REAL
    //==================================================

    real x_real;
    real d_real;
    real y_real;
    real e_real;

    //==================================================
    // MONITOR
    //==================================================

    always @(posedge clk) begin

        if (valid) begin

            y_real = $itor(y_out) / 32768.0;
            e_real = $itor(e_out) / 32768.0;

            $display("");
            $display("==============================================");
            $display(" RESULTADO");
            $display(" TIME = %0t ns", $time);
            $display(" Y    = %f", y_real);
            $display(" E    = %f", e_real);

            $display("");
            $display(" PESOS:");

            $display(
                " w0=%f | w1=%f | w2=%f | w3=%f",
                $itor(dut.u_datapath.w_mem[0]) / 32768.0,
                $itor(dut.u_datapath.w_mem[1]) / 32768.0,
                $itor(dut.u_datapath.w_mem[2]) / 32768.0,
                $itor(dut.u_datapath.w_mem[3]) / 32768.0
            );

            $display(
                " w4=%f | w5=%f | w6=%f | w7=%f",
                $itor(dut.u_datapath.w_mem[4]) / 32768.0,
                $itor(dut.u_datapath.w_mem[5]) / 32768.0,
                $itor(dut.u_datapath.w_mem[6]) / 32768.0,
                $itor(dut.u_datapath.w_mem[7]) / 32768.0
            );

            $display("");
            $display(" HISTORICO DE AMOSTRAS:");

            $display(
                " x0=%f | x1=%f | x2=%f | x3=%f",
                $itor(dut.u_datapath.x_mem[0]) / 32768.0,
                $itor(dut.u_datapath.x_mem[1]) / 32768.0,
                $itor(dut.u_datapath.x_mem[2]) / 32768.0,
                $itor(dut.u_datapath.x_mem[3]) / 32768.0
            );

            $display(
                " x4=%f | x5=%f | x6=%f | x7=%f",
                $itor(dut.u_datapath.x_mem[4]) / 32768.0,
                $itor(dut.u_datapath.x_mem[5]) / 32768.0,
                $itor(dut.u_datapath.x_mem[6]) / 32768.0,
                $itor(dut.u_datapath.x_mem[7]) / 32768.0
            );

            $display("==============================================");

        end

    end

    //==================================================
    // ENVIA UMA AMOSTRA
    //==================================================

    task send_sample;

        input signed [DATA_W-1:0] x_value;
        input signed [DATA_W-1:0] d_value;
        input integer sample_number;

        begin

            // Espera o módulo ficar livre
            wait (busy == 1'b0);

            // Coloca os dados antes do START
            @(negedge clk);

            x_in = x_value;
            d_in = d_value;

            // Conversão somente para visualizar
            x_real = $itor(x_value) / 32768.0;
            d_real = $itor(d_value) / 32768.0;

            $display("");
            $display("----------------------------------------------");
            $display(
                " ENVIANDO AMOSTRA %0d",
                sample_number
            );

            $display(
                " x_in = %f | d_in = %f",
                x_real,
                d_real
            );

            $display(
                " busy antes do start = %b",
                busy
            );

            $display("----------------------------------------------");

            // START
            start = 1'b1;

            @(negedge clk);

            start = 1'b0;

            // Agora o LMS está processando
            wait (valid == 1'b1);

            // Espera valid voltar a zero
            @(negedge clk);

            $display(
                " valid terminou em %b",
                valid
            );

        end

    endtask

    //==================================================
    // TESTE PRINCIPAL
    //==================================================

    initial begin

        // ---------------------------------------------
        // Inicialização
        // ---------------------------------------------

        reset = 1'b1;
        start = 1'b0;

        x_in = 16'sd0;
        d_in = 16'sd0;

        // ---------------------------------------------
        // VCD
        // ---------------------------------------------

        $dumpfile("lms_test2.vcd");
        $dumpvars(0, tb_lms);

        // ---------------------------------------------
        // RESET
        // ---------------------------------------------

        #100;

        reset = 1'b0;

        $display("");
        $display("==============================================");
        $display(" TESTE 2 - LMS COM AMOSTRAS VARIADAS");
        $display("==============================================");

        // ---------------------------------------------
        // AMOSTRA 1
        // x = +0,50
        // d = +0,25
        // ---------------------------------------------

        send_sample(
            16'sd16384,
            16'sd8192,
            1
        );

        // ---------------------------------------------
        // AMOSTRA 2
        // x = -0,50
        // d = -0,25
        // ---------------------------------------------

        send_sample(
            -16'sd16384,
            -16'sd8192,
            2
        );

        // ---------------------------------------------
        // AMOSTRA 3
        // x = +0,25
        // d = +0,125
        // ---------------------------------------------

        send_sample(
            16'sd8192,
            16'sd4096,
            3
        );

        // ---------------------------------------------
        // AMOSTRA 4
        // x = -0,25
        // d = -0,125
        // ---------------------------------------------

        send_sample(
            -16'sd8192,
            -16'sd4096,
            4
        );

        // ---------------------------------------------
        // AMOSTRA 5
        // x = +0,75
        // d = +0,375
        // ---------------------------------------------

        send_sample(
            16'sd24576,
            16'sd12288,
            5
        );

        // ---------------------------------------------
        // AMOSTRA 6
        // x = -0,75
        // d = -0,375
        // ---------------------------------------------

        send_sample(
            -16'sd24576,
            -16'sd12288,
            6
        );

        // ---------------------------------------------
        // AMOSTRA 7
        // x = +0,125
        // d = +0,0625
        // ---------------------------------------------

        send_sample(
            16'sd4096,
            16'sd2048,
            7
        );

        // ---------------------------------------------
        // AMOSTRA 8
        // x = -0,125
        // d = -0,0625
        // ---------------------------------------------

        send_sample(
            -16'sd4096,
            -16'sd2048,
            8
        );

        // ---------------------------------------------
        // AMOSTRA 9
        // x = +0,625
        // d = +0,3125
        // ---------------------------------------------

        send_sample(
            16'sd20480,
            16'sd10240,
            9
        );

        // ---------------------------------------------
        // AMOSTRA 10
        // x = -0,625
        // d = -0,3125
        // ---------------------------------------------

        send_sample(
            -16'sd20480,
            -16'sd10240,
            10
        );

        // ---------------------------------------------
        // FINAL
        // ---------------------------------------------

        $display("");
        $display("==============================================");
        $display(" TESTE 2 FINALIZADO");
        $display("==============================================");

        #100;

        $finish;

    end

endmodule