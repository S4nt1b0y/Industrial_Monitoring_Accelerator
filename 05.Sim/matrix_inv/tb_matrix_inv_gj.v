`timescale 1ns / 1ps

module tb_matrix_inv_gj;

localparam WORD_BITS = 16;
localparam VECTORS_BASE = "vectors/gj";
localparam CASES_PER_N  = 12;

integer total_errors;
integer total_cases;
reg [255:0] current_label;

initial begin
    #2000000;
    $display("GLOBAL WATCHDOG FIRED, stuck at: %0s", current_label);
    $finish;
end

function [255:0] category_name;
    input integer idx;
    begin
        if (idx < 4)
            category_name = "well_conditioned";
        else if (idx < 8)
            category_name = "near_singular";
        else
            category_name = "singular";
    end
endfunction

localparam N2 = 2;
reg                          clk2, rst_n2, start2;
reg  signed [N2*N2*WORD_BITS-1:0] a_in2;
reg  signed [N2*WORD_BITS-1:0]    b_in2;
wire                          busy2, done2, singular2;
wire signed [N2*WORD_BITS-1:0]    x_out2;
wire signed [N2*N2*WORD_BITS-1:0] ainv_out2;

matrix_inv_gj #(.N(N2), .WORD_BITS(WORD_BITS)) dut_n2 (
    .clk(clk2), .rst_n(rst_n2), .start(start2),
    .a_in(a_in2), .b_in(b_in2),
    .busy(busy2), .done(done2), .singular(singular2),
    .x_out(x_out2), .ainv_out(ainv_out2)
);
always #5 clk2 = ~clk2;

localparam N3 = 3;
reg                          clk3, rst_n3, start3;
reg  signed [N3*N3*WORD_BITS-1:0] a_in3;
reg  signed [N3*WORD_BITS-1:0]    b_in3;
wire                          busy3, done3, singular3;
wire signed [N3*WORD_BITS-1:0]    x_out3;
wire signed [N3*N3*WORD_BITS-1:0] ainv_out3;

matrix_inv_gj #(.N(N3), .WORD_BITS(WORD_BITS)) dut_n3 (
    .clk(clk3), .rst_n(rst_n3), .start(start3),
    .a_in(a_in3), .b_in(b_in3),
    .busy(busy3), .done(done3), .singular(singular3),
    .x_out(x_out3), .ainv_out(ainv_out3)
);
always #5 clk3 = ~clk3;

localparam N4 = 4;
reg                          clk4, rst_n4, start4;
reg  signed [N4*N4*WORD_BITS-1:0] a_in4;
reg  signed [N4*WORD_BITS-1:0]    b_in4;
wire                          busy4, done4, singular4;
wire signed [N4*WORD_BITS-1:0]    x_out4;
wire signed [N4*N4*WORD_BITS-1:0] ainv_out4;

matrix_inv_gj #(.N(N4), .WORD_BITS(WORD_BITS)) dut_n4 (
    .clk(clk4), .rst_n(rst_n4), .start(start4),
    .a_in(a_in4), .b_in(b_in4),
    .busy(busy4), .done(done4), .singular(singular4),
    .x_out(x_out4), .ainv_out(ainv_out4)
);
always #5 clk4 = ~clk4;

reg signed [WORD_BITS-1:0] a_mem2   [0:N2*N2-1];
reg signed [WORD_BITS-1:0] b_mem2   [0:N2-1];
reg signed [WORD_BITS-1:0] xexp_mem2   [0:N2-1];
reg signed [WORD_BITS-1:0] ainvexp_mem2 [0:N2*N2-1];
reg [7:0] sing_mem2 [0:0];

task run_case_n2;
    input integer idx;
    reg [255:0] cat;
    reg [256*8-1:0] path_buf;
    integer c;
    integer errors_here;
    reg expected_singular;
    begin
        cat = category_name(idx);
        $sformat(current_label, "n2 case_%02d_%0s", idx, cat);
        $sformat(path_buf, "%0s/n2/case_%02d_%0s/A.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, a_mem2);
        $sformat(path_buf, "%0s/n2/case_%02d_%0s/b.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, b_mem2);
        $sformat(path_buf, "%0s/n2/case_%02d_%0s/x_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, xexp_mem2);
        $sformat(path_buf, "%0s/n2/case_%02d_%0s/ainv_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, ainvexp_mem2);
        $sformat(path_buf, "%0s/n2/case_%02d_%0s/singular.txt", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, sing_mem2);
        expected_singular = sing_mem2[0][0];

        for (c = 0; c < N2*N2; c = c + 1)
            a_in2[(c+1)*WORD_BITS-1 -: WORD_BITS] = a_mem2[c];
        for (c = 0; c < N2; c = c + 1)
            b_in2[(c+1)*WORD_BITS-1 -: WORD_BITS] = b_mem2[c];

        @(negedge clk2);
        start2 = 1;
        @(negedge clk2);
        start2 = 0;
        wait (done2);

        errors_here = 0;
        if (singular2 !== expected_singular) begin
            errors_here = errors_here + 1;
            $display("FAIL n2 case_%0d (%0s): singular mismatch expected=%0d got=%0d",
                      idx, cat, expected_singular, singular2);
        end else if (!expected_singular) begin
            for (c = 0; c < N2; c = c + 1) begin
                if (x_out2[(c+1)*WORD_BITS-1 -: WORD_BITS] !== xexp_mem2[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n2 case_%0d x[%0d] expected=%0d got=%0d",
                              idx, c, xexp_mem2[c], x_out2[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
            for (c = 0; c < N2*N2; c = c + 1) begin
                if (ainv_out2[(c+1)*WORD_BITS-1 -: WORD_BITS] !== ainvexp_mem2[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n2 case_%0d ainv[%0d] expected=%0d got=%0d",
                              idx, c, ainvexp_mem2[c], ainv_out2[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
        end
        if (errors_here == 0)
            $display("PASS n2 case_%0d (%0s)", idx, cat);
        total_errors = total_errors + errors_here;
        total_cases  = total_cases + 1;
        @(negedge clk2);
    end
endtask

reg signed [WORD_BITS-1:0] a_mem3   [0:N3*N3-1];
reg signed [WORD_BITS-1:0] b_mem3   [0:N3-1];
reg signed [WORD_BITS-1:0] xexp_mem3   [0:N3-1];
reg signed [WORD_BITS-1:0] ainvexp_mem3 [0:N3*N3-1];
reg [7:0] sing_mem3 [0:0];

task run_case_n3;
    input integer idx;
    reg [255:0] cat;
    reg [256*8-1:0] path_buf;
    integer c;
    integer errors_here;
    reg expected_singular;
    begin
        cat = category_name(idx);
        $sformat(current_label, "n3 case_%02d_%0s", idx, cat);
        $sformat(path_buf, "%0s/n3/case_%02d_%0s/A.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, a_mem3);
        $sformat(path_buf, "%0s/n3/case_%02d_%0s/b.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, b_mem3);
        $sformat(path_buf, "%0s/n3/case_%02d_%0s/x_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, xexp_mem3);
        $sformat(path_buf, "%0s/n3/case_%02d_%0s/ainv_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, ainvexp_mem3);
        $sformat(path_buf, "%0s/n3/case_%02d_%0s/singular.txt", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, sing_mem3);
        expected_singular = sing_mem3[0][0];

        for (c = 0; c < N3*N3; c = c + 1)
            a_in3[(c+1)*WORD_BITS-1 -: WORD_BITS] = a_mem3[c];
        for (c = 0; c < N3; c = c + 1)
            b_in3[(c+1)*WORD_BITS-1 -: WORD_BITS] = b_mem3[c];

        @(negedge clk3);
        start3 = 1;
        @(negedge clk3);
        start3 = 0;
        wait (done3);

        errors_here = 0;
        if (singular3 !== expected_singular) begin
            errors_here = errors_here + 1;
            $display("FAIL n3 case_%0d (%0s): singular mismatch expected=%0d got=%0d",
                      idx, cat, expected_singular, singular3);
        end else if (!expected_singular) begin
            for (c = 0; c < N3; c = c + 1) begin
                if (x_out3[(c+1)*WORD_BITS-1 -: WORD_BITS] !== xexp_mem3[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n3 case_%0d x[%0d] expected=%0d got=%0d",
                              idx, c, xexp_mem3[c], x_out3[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
            for (c = 0; c < N3*N3; c = c + 1) begin
                if (ainv_out3[(c+1)*WORD_BITS-1 -: WORD_BITS] !== ainvexp_mem3[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n3 case_%0d ainv[%0d] expected=%0d got=%0d",
                              idx, c, ainvexp_mem3[c], ainv_out3[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
        end
        if (errors_here == 0)
            $display("PASS n3 case_%0d (%0s)", idx, cat);
        total_errors = total_errors + errors_here;
        total_cases  = total_cases + 1;
        @(negedge clk3);
    end
endtask

reg signed [WORD_BITS-1:0] a_mem4   [0:N4*N4-1];
reg signed [WORD_BITS-1:0] b_mem4   [0:N4-1];
reg signed [WORD_BITS-1:0] xexp_mem4   [0:N4-1];
reg signed [WORD_BITS-1:0] ainvexp_mem4 [0:N4*N4-1];
reg [7:0] sing_mem4 [0:0];

task run_case_n4;
    input integer idx;
    reg [255:0] cat;
    reg [256*8-1:0] path_buf;
    integer c;
    integer errors_here;
    reg expected_singular;
    begin
        cat = category_name(idx);
        $sformat(current_label, "n4 case_%02d_%0s", idx, cat);
        $sformat(path_buf, "%0s/n4/case_%02d_%0s/A.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, a_mem4);
        $sformat(path_buf, "%0s/n4/case_%02d_%0s/b.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, b_mem4);
        $sformat(path_buf, "%0s/n4/case_%02d_%0s/x_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, xexp_mem4);
        $sformat(path_buf, "%0s/n4/case_%02d_%0s/ainv_expected.hex", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, ainvexp_mem4);
        $sformat(path_buf, "%0s/n4/case_%02d_%0s/singular.txt", VECTORS_BASE, idx, cat);
        $readmemh(path_buf, sing_mem4);
        expected_singular = sing_mem4[0][0];

        for (c = 0; c < N4*N4; c = c + 1)
            a_in4[(c+1)*WORD_BITS-1 -: WORD_BITS] = a_mem4[c];
        for (c = 0; c < N4; c = c + 1)
            b_in4[(c+1)*WORD_BITS-1 -: WORD_BITS] = b_mem4[c];

        @(negedge clk4);
        start4 = 1;
        @(negedge clk4);
        start4 = 0;
        wait (done4);

        errors_here = 0;
        if (singular4 !== expected_singular) begin
            errors_here = errors_here + 1;
            $display("FAIL n4 case_%0d (%0s): singular mismatch expected=%0d got=%0d",
                      idx, cat, expected_singular, singular4);
        end else if (!expected_singular) begin
            for (c = 0; c < N4; c = c + 1) begin
                if (x_out4[(c+1)*WORD_BITS-1 -: WORD_BITS] !== xexp_mem4[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n4 case_%0d x[%0d] expected=%0d got=%0d",
                              idx, c, xexp_mem4[c], x_out4[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
            for (c = 0; c < N4*N4; c = c + 1) begin
                if (ainv_out4[(c+1)*WORD_BITS-1 -: WORD_BITS] !== ainvexp_mem4[c]) begin
                    errors_here = errors_here + 1;
                    $display("FAIL n4 case_%0d ainv[%0d] expected=%0d got=%0d",
                              idx, c, ainvexp_mem4[c], ainv_out4[(c+1)*WORD_BITS-1 -: WORD_BITS]);
                end
            end
        end
        if (errors_here == 0)
            $display("PASS n4 case_%0d (%0s)", idx, cat);
        total_errors = total_errors + errors_here;
        total_cases  = total_cases + 1;
        @(negedge clk4);
    end
endtask

integer case_idx;

initial begin
    clk2 = 0; rst_n2 = 0; start2 = 0; a_in2 = 0; b_in2 = 0;
    clk3 = 0; rst_n3 = 0; start3 = 0; a_in3 = 0; b_in3 = 0;
    clk4 = 0; rst_n4 = 0; start4 = 0; a_in4 = 0; b_in4 = 0;
    total_errors = 0;
    total_cases  = 0;

    #12;
    rst_n2 = 1; rst_n3 = 1; rst_n4 = 1;
    #10;

    for (case_idx = 0; case_idx < CASES_PER_N; case_idx = case_idx + 1)
        run_case_n2(case_idx);
    for (case_idx = 0; case_idx < CASES_PER_N; case_idx = case_idx + 1)
        run_case_n3(case_idx);
    for (case_idx = 0; case_idx < CASES_PER_N; case_idx = case_idx + 1)
        run_case_n4(case_idx);

    if (total_errors == 0)
        $display("ALL %0d MATRIX_INV_GJ CASES PASSED", total_cases);
    else
        $display("%0d/%0d MATRIX_INV_GJ CASES FAILED", total_errors, total_cases);

    $finish;
end

endmodule
