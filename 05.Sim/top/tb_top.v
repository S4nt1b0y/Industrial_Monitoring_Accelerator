`timescale 1ns/1ps

module tb_top;

localparam DATA_WIDTH = 16;
localparam N = 64;

reg clk;
reg rst_n;
reg valid_i;
reg signed [N*DATA_WIDTH-1:0] acc_x_a_i;
reg signed [N*DATA_WIDTH-1:0] acc_x_b_i;
reg signed [N*DATA_WIDTH-1:0] acc_y_a_i;
reg signed [N*DATA_WIDTH-1:0] acc_y_b_i;
wire ready_o;
wire valid_o;
wire [1:0] class_o;

integer i;
integer cycles;
integer fft_starts;
integer mdc_starts;
integer failures;

ml_pipeline #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N),
    .M(6),
    .FRACW(15)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .acc_x_a_i(acc_x_a_i),
    .acc_x_b_i(acc_x_b_i),
    .acc_y_a_i(acc_y_a_i),
    .acc_y_b_i(acc_y_b_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .valid_o(valid_o),
    .class_o(class_o)
);

always #5 clk = ~clk;

task set_sample;
    inout signed [N*DATA_WIDTH-1:0] channel;
    input integer index;
    input signed [DATA_WIDTH-1:0] value;
    begin
        channel[(index+1)*DATA_WIDTH-1 -: DATA_WIDTH] = value;
    end
endtask

task load_channels;
    begin
        acc_x_a_i = {N*DATA_WIDTH{1'b0}};
        acc_x_b_i = {N*DATA_WIDTH{1'b0}};
        acc_y_a_i = {N*DATA_WIDTH{1'b0}};
        acc_y_b_i = {N*DATA_WIDTH{1'b0}};

        for (i = 0; i < N; i = i + 1) begin
            set_sample(acc_x_a_i, i, i);
            set_sample(acc_x_b_i, i, i + 16);
            set_sample(acc_y_a_i, i, i + 32);
            set_sample(acc_y_b_i, i, i + 48);
        end
    end
endtask

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    valid_i = 1'b0;
    failures = 0;
    fft_starts = 0;
    mdc_starts = 0;
    load_channels();

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    #1;

    if (ready_o !== 1'b1) begin
        $display("FAIL: ready_o should be high in IDLE after reset");
        failures = failures + 1;
    end

    @(negedge clk);
    valid_i = 1'b1;
    @(posedge clk);
    #1;

    if (ready_o !== 1'b0) begin
        $display("FAIL: ready_o should drop after accepting valid_i");
        failures = failures + 1;
    end

    @(negedge clk);
    valid_i = 1'b0;

    cycles = 0;
    while (!valid_o && cycles < 20000) begin
        @(posedge clk);
        if (dut.fft_start) begin
            fft_starts = fft_starts + 1;
        end
        if (dut.mdc_start) begin
            mdc_starts = mdc_starts + 1;
        end
        if (ready_o) begin
            $display("FAIL: ready_o returned high before valid_o");
            failures = failures + 1;
        end
        cycles = cycles + 1;
    end

    if (!valid_o) begin
        $display("FAIL: pipeline did not assert valid_o before timeout");
        failures = failures + 1;
    end

    @(posedge clk);
    #1;

    if (ready_o !== 1'b1) begin
        $display("FAIL: ready_o should return high after classification");
        failures = failures + 1;
    end

    if (fft_starts != 4) begin
        $display("FAIL: expected 4 FFT starts, got %0d", fft_starts);
        failures = failures + 1;
    end

    if (mdc_starts != 4) begin
        $display("FAIL: expected 4 MDC starts, got %0d", mdc_starts);
        failures = failures + 1;
    end

    if (^dut.features[2239:2112] === 1'bx) begin
        $display("FAIL: MDC feature slice contains unknown values");
        failures = failures + 1;
    end

    if (failures == 0) begin
        $display("All ml_pipeline handshake tests passed. class_o=%0d", class_o);
    end else begin
        $display("%0d ml_pipeline test(s) failed.", failures);
    end

    $finish;
end

endmodule
