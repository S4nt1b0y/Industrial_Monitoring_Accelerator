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
wire signed [N*DATA_WIDTH-1:0] sample_block_o;
wire [1:0] channel_o;
wire overflow_o;

integer failures;
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
    .sample_block_o(sample_block_o),
    .channel_o(channel_o),
    .overflow_o(overflow_o)
);

always #5 clk = ~clk;

function [DATA_WIDTH-1:0] get_sample;
    input integer idx;
    begin
        get_sample = sample_block_o[(idx+1)*DATA_WIDTH-1 -: DATA_WIDTH];
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

task send_block;
    input integer channel;
    begin
        for (sample = 0; sample < N; sample = sample + 1) begin
            send_sample((channel << 8) + sample);
        end
    end
endtask

task check_visible_block;
    input integer channel;
    begin
        @(posedge clk);
        #1;

        if (frame_valid_o !== 1'b1) begin
            $display("FAIL: frame_valid_o should be high for channel %0d", channel);
            failures = failures + 1;
        end

        if (channel_o !== channel[1:0]) begin
            $display("FAIL: expected visible channel %0d got %0d", channel, channel_o);
            failures = failures + 1;
        end

        for (sample = 0; sample < N; sample = sample + 1) begin
            expected = (channel << 8) + sample;
            if (get_sample(sample) !== expected) begin
                $display("FAIL: channel %0d sample %0d expected %h got %h",
                         channel, sample, expected, get_sample(sample));
                failures = failures + 1;
            end
        end
    end
endtask

task accept_visible_block;
    begin
        @(negedge clk);
        frame_ready_i = 1'b1;
        @(posedge clk);
        #1;
        frame_ready_i = 1'b0;
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

    send_block(0);
    check_visible_block(0);

    send_block(1);
    check_visible_block(0);

    accept_visible_block();
    check_visible_block(1);

    send_block(2);
    check_visible_block(1);

    send_byte(8'hAA);
    @(posedge clk);
    #1;

    if (overflow_o !== 1'b1) begin
        $display("FAIL: overflow_o should assert when both ping-pong banks are occupied");
        failures = failures + 1;
    end

    accept_visible_block();
    accept_visible_block();

    if (frame_valid_o !== 1'b0) begin
        $display("FAIL: frame_valid_o should clear after both pending blocks are accepted");
        failures = failures + 1;
    end

    if (overflow_o !== 1'b0) begin
        $display("FAIL: overflow_o should clear after accepting a pending block");
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
