`timescale 1ns/1ps

module tb_fft;
    localparam N     = 64;
    localparam M     = 6;
    localparam W     = 16;
    localparam FRACW = 15;
    localparam NUM_WINDOWS = 10;
    localparam TOTAL_SAMPLES = N * NUM_WINDOWS;

    reg clk;
    reg rst_n;
    reg start;
    reg signed [N*W-1:0] in_re;
    reg signed [N*W-1:0] in_im;
    wire done;
    wire signed [N*W-1:0] out_re;
    wire signed [N*W-1:0] out_im;

    reg signed [W-1:0] input_re [0:TOTAL_SAMPLES-1];
    reg signed [W-1:0] input_im [0:TOTAL_SAMPLES-1];

    integer i;
    integer window;
    integer sample;
    integer input_fd;
    integer output_fd;
    integer fields;
    integer file_window;
    integer file_sample;
    integer file_re;
    integer file_im;
    integer cycles;

    fftu_dif #(
        .N(N),
        .M(M),
        .W(W),
        .FRACW(FRACW),
        .BF_LAT(2)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_re(in_re),
        .in_im(in_im),
        .done(done),
        .out_re(out_re),
        .out_im(out_im)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task load_window;
        input integer win;
        begin
            for (sample = 0; sample < N; sample = sample + 1) begin
                in_re[(sample+1)*W-1 -: W] = input_re[win*N + sample];
                in_im[(sample+1)*W-1 -: W] = input_im[win*N + sample];
            end
        end
    endtask

    task start_and_wait_done;
        begin
            cycles = 0;
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;

            while (!done && cycles < 2000) begin
                @(posedge clk);
                cycles = cycles + 1;
            end

            if (!done) begin
                $display("ERROR: FFT did not assert done before timeout");
                $finish;
            end
        end
    endtask

    task write_window_output;
        input integer win;
        begin
            for (sample = 0; sample < N; sample = sample + 1) begin
                $fdisplay(output_fd, "%0d %0d %0d %0d",
                    win,
                    sample,
                    $signed(out_re[(sample+1)*W-1 -: W]),
                    $signed(out_im[(sample+1)*W-1 -: W])
                );
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        in_re = {N*W{1'b0}};
        in_im = {N*W{1'b0}};
        input_fd = $fopen("input.txt", "r");
        if (input_fd == 0) begin
            $display("ERROR: could not open input.txt");
            $finish;
        end

        for (i = 0; i < TOTAL_SAMPLES; i = i + 1) begin
            fields = $fscanf(input_fd, "%d %d %d %d\n",
                file_window, file_sample, file_re, file_im);

            if (fields != 4) begin
                $display("ERROR: malformed input.txt line %0d", i + 1);
                $finish;
            end

            if (file_window != (i / N) || file_sample != (i % N)) begin
                $display("ERROR: unexpected input index at line %0d", i + 1);
                $finish;
            end

            input_re[i] = file_re[W-1:0];
            input_im[i] = file_im[W-1:0];
        end

        $fclose(input_fd);

        output_fd = $fopen("rtl_output.txt", "w");
        if (output_fd == 0) begin
            $display("ERROR: could not open rtl_output.txt");
            $finish;
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        for (window = 0; window < NUM_WINDOWS; window = window + 1) begin
            load_window(window);
            start_and_wait_done();
            write_window_output(window);
            @(posedge clk);
        end

        $fclose(output_fd);
        $display("RTL FFT output written to rtl_output.txt");
        $finish;
    end
endmodule
