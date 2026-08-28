`timescale 1ns/1ps

module tb_mdc_tres_picos;

    localparam WIDTH = 6;
    localparam N_FFT = 64;

    reg clk;
    reg reset;
    reg start;

    reg  [WIDTH-1:0] pico_in;
    reg              pico_valid;
    wire             pico_ready;

    reg  [31:0]      fs_hz;
    reg  [WIDTH-1:0] min_k;

    wire [WIDTH-1:0] k0;
    wire [31:0]      f0;
    wire             busy;
    wire             done;
    wire             result_valid;

    integer erros;


    mdc_tres_picos #(
        .WIDTH(WIDTH),
        .N_FFT(N_FFT)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .pico_in(pico_in),
        .pico_valid(pico_valid),
        .pico_ready(pico_ready),
        .fs_hz(fs_hz),
        .min_k(min_k),
        .k0(k0),
        .f0(f0),
        .busy(busy),
        .done(done),
        .result_valid(result_valid)
    );


    // clock de 100 MHz
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end


    // Envia um pico somente quando o módulo estiver pronto
    task envia_pico;
        input [WIDTH-1:0] valor;
        begin
            wait (pico_ready);

            @(negedge clk);
            pico_in    = valor;
            pico_valid = 1;

            @(negedge clk);
            pico_valid = 0;
        end
    endtask


    // Executa uma janela completa com três picos
    task testa_janela;
        input [WIDTH-1:0] p1;
        input [WIDTH-1:0] p2;
        input [WIDTH-1:0] p3;

        input [WIDTH-1:0] k0_esperado;
        input [31:0]      f0_esperado;
        input              valid_esperado;

        begin
            $display("");
            $display("Testando picos: %0d, %0d, %0d", p1, p2, p3);

            // inicia uma nova janela
            @(negedge clk);
            start = 1;

            @(negedge clk);
            start = 0;

            envia_pico(p1);
            envia_pico(p2);
            envia_pico(p3);

            // espera o processamento terminar
            wait (done);
            #1;

            $display("Obtido   -> k0 = %0d | f0 = %0d Hz | valid = %0d",
                     k0, f0, result_valid);

            $display("Esperado -> k0 = %0d | f0 = %0d Hz | valid = %0d",
                     k0_esperado, f0_esperado, valid_esperado);

            if ((k0 == k0_esperado) &&
                (f0 == f0_esperado) &&
                (result_valid == valid_esperado)) begin

                $display("Resultado: OK");

            end
            else begin
                $display("Resultado: ERRO");
                erros = erros + 1;
            end

            // deixa o módulo voltar ao IDLE antes do próximo teste
            @(posedge clk);
            @(posedge clk);
        end
    endtask


    initial begin
        reset      = 1;
        start      = 0;
        pico_in    = 0;
        pico_valid = 0;

        fs_hz = 6400;
        min_k = 2;

        erros = 0;

        // reset inicial
        repeat (2) @(posedge clk);
        reset = 0;


        // Mesmos casos usados no modelo em Python

        // MDC(12,18,30) = 6
        // f0 = 6 * 6400 / 64 = 600 Hz
        testa_janela(12, 18, 30, 6, 600, 1);

        // MDC(8,16,24) = 8
        testa_janela(8, 16, 24, 8, 800, 1);

        // MDC(5,10,20) = 5
        testa_janela(5, 10, 20, 5, 500, 1);

        // Pico igual a zero: janela inválida
        testa_janela(0, 18, 30, 0, 0, 0);

        // Três picos iguais
        testa_janela(12, 12, 12, 12, 1200, 1);

        // MDC = 1, menor que min_k = 2: janela inválida
        testa_janela(12, 17, 31, 0, 0, 0);


        $display("");
        $display("----------------------------------------");

        if (erros == 0)
            $display("Todos os testes passaram.");
        else
            $display("Foram encontrados %0d erro(s).", erros);

        $display("----------------------------------------");

        #20;
        $finish;
    end

endmodule
