`timescale 1ns/1ps

module tb_fft;
    localparam N     = 64;
    localparam M     = 6;
    localparam W     = 16;
    localparam FRACW = 15;

    reg clk;
    reg rst_n;
    reg start;
    reg signed [N*W-1:0] in_re;
    reg signed [N*W-1:0] in_im;
    wire done;
    wire signed [N*W-1:0] out_re;
    wire signed [N*W-1:0] out_im;

    integer i;
    integer fd;
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

    function signed [W-1:0] sample_re;
        input integer idx;
        begin
            sample_re = ((idx * 1103 + 12345) % 24576) - 12288;
        end
    endfunction

    function signed [W-1:0] sample_im;
        input integer idx;
        begin
            sample_im = ((idx * 1877 + 5432) % 16384) - 8192;
        end
    endfunction

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        in_re = {N*W{1'b0}};
        in_im = {N*W{1'b0}};
        cycles = 0;

        for (i = 0; i < N; i = i + 1) begin
            in_re[(i+1)*W-1 -: W] = sample_re(i);
            in_im[(i+1)*W-1 -: W] = sample_im(i);
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

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

        fd = $fopen("rtl_output.txt", "w");
        if (fd == 0) begin
            $display("ERROR: could not open rtl_output.txt");
            $finish;
        end

        for (i = 0; i < N; i = i + 1) begin
            $fdisplay(fd, "%0d %0d %0d",
                i,
                $signed(out_re[(i+1)*W-1 -: W]),
                $signed(out_im[(i+1)*W-1 -: W])
            );
        end

        $fclose(fd);
        $display("RTL FFT output written to rtl_output.txt");
        $finish;
    end
endmodule
