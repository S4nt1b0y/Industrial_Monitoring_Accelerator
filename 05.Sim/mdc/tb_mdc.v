`timescale 1ns/1ps

module tb_mdc;

localparam WIDTH = 6;
localparam N_FFT = 64;

reg clk;
reg reset;
reg start;
reg [WIDTH-1:0] pico1_i;
reg [WIDTH-1:0] pico2_i;
reg [WIDTH-1:0] pico3_i;
reg [31:0] fs_hz;
reg [WIDTH-1:0] min_k;

wire [WIDTH-1:0] k0;
wire [31:0] f0;
wire busy;
wire done;
wire result_valid;

integer erros;
integer cycles;

mdc_tres_picos #(
    .WIDTH(WIDTH),
    .N_FFT(N_FFT)
) dut (
    .clk(clk),
    .reset(reset),
    .start(start),
    .pico1_i(pico1_i),
    .pico2_i(pico2_i),
    .pico3_i(pico3_i),
    .fs_hz(fs_hz),
    .min_k(min_k),
    .k0(k0),
    .f0(f0),
    .busy(busy),
    .done(done),
    .result_valid(result_valid)
);

initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
end

task testa_janela;
    input [WIDTH-1:0] p1;
    input [WIDTH-1:0] p2;
    input [WIDTH-1:0] p3;
    input [WIDTH-1:0] k0_esperado;
    input [31:0] f0_esperado;
    input valid_esperado;
    begin
        @(negedge clk);
        pico1_i = p1;
        pico2_i = p2;
        pico3_i = p3;
        start = 1'b1;

        @(negedge clk);
        start = 1'b0;

        fork : wait_done_or_timeout
            begin
                @(posedge done);
            end
            begin
                cycles = 0;
                while (!done && cycles < 1000) begin
                    @(posedge clk);
                    cycles = cycles + 1;
                end
            end
        join_any
        disable wait_done_or_timeout;

        if (!done) begin
            $display("FAIL timeout for peaks %0d, %0d, %0d", p1, p2, p3);
            erros = erros + 1;
        end else begin
            #1;
            if ((k0 != k0_esperado) ||
                (f0 != f0_esperado) ||
                (result_valid != valid_esperado)) begin
                $display("FAIL peaks %0d, %0d, %0d: k0=%0d f0=%0d valid=%0d",
                         p1, p2, p3, k0, f0, result_valid);
                erros = erros + 1;
            end else begin
                $display("PASS peaks %0d, %0d, %0d", p1, p2, p3);
            end
        end

        @(posedge clk);
        @(posedge clk);
    end
endtask

initial begin
    reset = 1'b1;
    start = 1'b0;
    pico1_i = {WIDTH{1'b0}};
    pico2_i = {WIDTH{1'b0}};
    pico3_i = {WIDTH{1'b0}};
    fs_hz = 6400;
    min_k = 2;
    erros = 0;

    repeat (2) @(posedge clk);
    reset = 1'b0;

    testa_janela(12, 18, 30, 6, 600, 1);
    testa_janela(8, 16, 24, 8, 800, 1);
    testa_janela(5, 10, 20, 5, 500, 1);
    testa_janela(0, 18, 30, 0, 0, 0);
    testa_janela(12, 12, 12, 12, 1200, 1);
    testa_janela(12, 17, 31, 0, 0, 0);

    if (erros == 0) begin
        $display("All MDC tests passed.");
    end else begin
        $display("%0d MDC test(s) failed.", erros);
    end

    #20;
    $finish;
end

endmodule
