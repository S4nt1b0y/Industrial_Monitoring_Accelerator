`timescale 1ns/1ps

module tb_top;

localparam DATA_WIDTH = 16;
localparam N = 64;

reg clk;
reg rst_n;
reg valid_i;
reg signed [N*DATA_WIDTH-1:0] sample_block_i;
reg [1:0] channel_i;
wire ready_o;
wire valid_o;
wire [1:0] class_o;

integer i;
integer channel;
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
    .sample_block_i(sample_block_i),
    .channel_i(channel_i),
    .valid_i(valid_i),
    .ready_o(ready_o),
    .valid_o(valid_o),
    .class_o(class_o)
);

always #5 clk = ~clk;

task set_sample;
    input integer index;
    input signed [DATA_WIDTH-1:0] value;
    begin
        sample_block_i[(index+1)*DATA_WIDTH-1 -: DATA_WIDTH] = value;
    end
endtask

task load_block;
    input integer ch;
    begin
        sample_block_i = {N*DATA_WIDTH{1'b0}};
        for (i = 0; i < N; i = i + 1) begin
            set_sample(i, i + (ch * 16));
        end
    end
endtask

task send_block;
    input integer ch;
    begin
        cycles = 0;
        while (!ready_o && cycles < 20000) begin
            @(posedge clk);
            cycles = cycles + 1;
        end

        if (!ready_o) begin
            $display("FAIL: pipeline did not become ready before channel %0d", ch);
            failures = failures + 1;
        end

        load_block(ch);
        @(negedge clk);
        channel_i = ch[1:0];
        valid_i = 1'b1;
        @(posedge clk);
        #1;
        valid_i = 1'b0;

        if (ready_o !== 1'b0) begin
            $display("FAIL: ready_o should drop after accepting channel %0d", ch);
            failures = failures + 1;
        end
    end
endtask

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    valid_i = 1'b0;
    channel_i = 2'd0;
    sample_block_i = {N*DATA_WIDTH{1'b0}};
    failures = 0;
    fft_starts = 0;
    mdc_starts = 0;

    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);
    #1;

    if (ready_o !== 1'b1) begin
        $display("FAIL: ready_o should be high in IDLE after reset");
        failures = failures + 1;
    end

    for (channel = 0; channel < 4; channel = channel + 1) begin
        send_block(channel);

        cycles = 0;
        while (!ready_o && !valid_o && cycles < 20000) begin
            @(posedge clk);
            if (dut.fft_start) begin
                fft_starts = fft_starts + 1;
            end
            if (dut.mdc_start) begin
                mdc_starts = mdc_starts + 1;
            end
            cycles = cycles + 1;
        end

        if (channel < 3) begin
            if (valid_o) begin
                $display("FAIL: valid_o asserted before channel 3 completed");
                failures = failures + 1;
            end
            if (!ready_o) begin
                $display("FAIL: pipeline did not return ready after channel %0d", channel);
                failures = failures + 1;
            end
        end else begin
            if (!valid_o) begin
                $display("FAIL: pipeline did not assert valid_o after channel 3");
                failures = failures + 1;
            end
        end
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
