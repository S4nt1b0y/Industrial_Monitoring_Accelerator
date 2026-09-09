`timescale 1ns/1ps

module tb_uart_frame_buffer;

localparam DATA_WIDTH = 16;
localparam N = 64;

reg clk;
reg rst_n;
reg [7:0] byte_i;
reg byte_valid_i;
reg frame_ready_i;
wire frame_valid_o;
wire signed [N*DATA_WIDTH-1:0] acc_x_a_o;
wire signed [N*DATA_WIDTH-1:0] acc_x_b_o;
wire signed [N*DATA_WIDTH-1:0] acc_y_a_o;
wire signed [N*DATA_WIDTH-1:0] acc_y_b_o;
wire overflow_o;

integer failures;
integer channel;
integer sample;
reg [15:0] expected;

uart_frame_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .byte_i(byte_i),
    .byte_valid_i(byte_valid_i),
    .frame_ready_i(frame_ready_i),
    .frame_valid_o(frame_valid_o),
    .acc_x_a_o(acc_x_a_o),
    .acc_x_b_o(acc_x_b_o),
    .acc_y_a_o(acc_y_a_o),
    .acc_y_b_o(acc_y_b_o),
    .overflow_o(overflow_o)
);

always #5 clk = ~clk;

function [DATA_WIDTH-1:0] get_sample;
    input integer ch;
    input integer idx;
    begin
        case (ch)
            0: get_sample = acc_x_a_o[(idx+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            1: get_sample = acc_x_b_o[(idx+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            2: get_sample = acc_y_a_o[(idx+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            3: get_sample = acc_y_b_o[(idx+1)*DATA_WIDTH-1 -: DATA_WIDTH];
            default: get_sample = {DATA_WIDTH{1'b0}};
        endcase
    end
endfunction

task send_byte;
    input [7:0] value;
    begin
        @(negedge clk);
        byte_i = value;
        byte_valid_i = 1'b1;
        @(negedge clk);
        byte_valid_i = 1'b0;
    end
endtask

task send_sample;
    input [15:0] value;
    begin
        send_byte(value[15:8]);
        send_byte(value[7:0]);
    end
endtask

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    byte_i = 8'd0;
    byte_valid_i = 1'b0;
    frame_ready_i = 1'b0;
    failures = 0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;

    for (channel = 0; channel < 4; channel = channel + 1) begin
        for (sample = 0; sample < N; sample = sample + 1) begin
            send_sample((channel << 8) + sample);
        end
    end

    @(posedge clk);
    #1;

    if (frame_valid_o !== 1'b1) begin
        $display("FAIL: frame_valid_o should be held after 256 samples");
        failures = failures + 1;
    end

    for (channel = 0; channel < 4; channel = channel + 1) begin
        for (sample = 0; sample < N; sample = sample + 1) begin
            expected = (channel << 8) + sample;
            if (get_sample(channel, sample) !== expected) begin
                $display("FAIL: channel %0d sample %0d expected %h got %h",
                         channel, sample, expected, get_sample(channel, sample));
                failures = failures + 1;
            end
        end
    end

    send_byte(8'hAA);
    @(posedge clk);
    #1;

    if (overflow_o !== 1'b1) begin
        $display("FAIL: overflow_o should assert when bytes arrive with a pending frame");
        failures = failures + 1;
    end

    @(negedge clk);
    frame_ready_i = 1'b1;
    @(posedge clk);
    #1;
    frame_ready_i = 1'b0;

    if (frame_valid_o !== 1'b0) begin
        $display("FAIL: frame_valid_o should clear after frame_ready_i");
        failures = failures + 1;
    end

    if (overflow_o !== 1'b0) begin
        $display("FAIL: overflow_o should clear after frame is accepted");
        failures = failures + 1;
    end

    if (failures == 0) begin
        $display("All uart_frame_buffer tests passed.");
    end else begin
        $display("%0d uart_frame_buffer test(s) failed.", failures);
    end

    $finish;
end

endmodule
