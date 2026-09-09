`timescale 1ns/1ps

module tb_top_wrapper;

localparam CLK_FREQ_HZ = 1000000;
localparam BAUD_DIV = 104;

reg clk;
reg rst_n;
reg ml_swith;
reg uart_rx_i;
wire Led_Normal;
wire Led_Unbalaced;
wire Led_disalaighn;
wire Led_desgaste;

integer failures;
integer channel;
integer sample;
reg seen_cnn_valid;
reg seen_ml_valid;

top_wrapper #(
    .CLK_FREQ_HZ(CLK_FREQ_HZ)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .ml_swith(ml_swith),
    .uart_rx_i(uart_rx_i),
    .Led_Normal(Led_Normal),
    .Led_Unbalaced(Led_Unbalaced),
    .Led_disalaighn(Led_disalaighn),
    .Led_desgaste(Led_desgaste)
);

always #5 clk = ~clk;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        seen_cnn_valid <= 1'b0;
        seen_ml_valid  <= 1'b0;
    end else begin
        if (dut.u_cnn.valid_o) begin
            seen_cnn_valid <= 1'b1;
        end
        if (dut.u_ml_pipeline.valid_i) begin
            seen_ml_valid <= 1'b1;
        end
    end
end

task uart_send_byte;
    input [7:0] value;
    integer bit_idx;
    begin
        uart_rx_i = 1'b0;
        repeat (BAUD_DIV) @(posedge clk);
        for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
            uart_rx_i = value[bit_idx];
            repeat (BAUD_DIV) @(posedge clk);
        end
        uart_rx_i = 1'b1;
        repeat (BAUD_DIV) @(posedge clk);
    end
endtask

task uart_send_sample;
    input [15:0] value;
    begin
        uart_send_byte(value[15:8]);
        uart_send_byte(value[7:0]);
    end
endtask

task uart_send_frame;
    begin
        for (channel = 0; channel < 4; channel = channel + 1) begin
            for (sample = 0; sample < 64; sample = sample + 1) begin
                uart_send_sample((channel << 8) + sample);
            end
        end
    end
endtask

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    ml_swith = 1'b1;
    uart_rx_i = 1'b1;
    failures = 0;

    repeat (8) @(posedge clk);
    rst_n = 1'b1;
    repeat (8) @(posedge clk);

    uart_send_frame();

    repeat (4) @(posedge clk);
    #1;

    if (seen_cnn_valid !== 1'b1) begin
        $display("FAIL: CNN path should produce valid_o for ml_swith=1");
        failures = failures + 1;
    end

    if (dut.u_ml_pipeline.valid_i !== 1'b0) begin
        $display("FAIL: ML pipeline should not receive valid_i when ml_swith=1");
        failures = failures + 1;
    end

    if (Led_Normal !== 1'b1 ||
        Led_Unbalaced !== 1'b0 ||
        Led_disalaighn !== 1'b0 ||
        Led_desgaste !== 1'b0) begin
        $display("FAIL: expected only Led_Normal after CNN fake classification");
        failures = failures + 1;
    end

    rst_n = 1'b0;
    ml_swith = 1'b0;
    uart_rx_i = 1'b1;
    repeat (8) @(posedge clk);
    rst_n = 1'b1;
    repeat (8) @(posedge clk);

    uart_send_frame();
    repeat (4) @(posedge clk);
    #1;

    if (seen_ml_valid !== 1'b1) begin
        $display("FAIL: ML pipeline should receive valid_i for ml_swith=0");
        failures = failures + 1;
    end

    if (dut.u_cnn.valid_i !== 1'b0) begin
        $display("FAIL: CNN should not receive valid_i when ml_swith=0");
        failures = failures + 1;
    end

    if (failures == 0) begin
        $display("All top_wrapper UART/CNN route tests passed.");
    end else begin
        $display("%0d top_wrapper test(s) failed.", failures);
    end

    $finish;
end

endmodule
