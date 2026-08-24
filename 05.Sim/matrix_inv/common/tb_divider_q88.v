
`timescale 1ns / 1ps

module tb_divider_q88;

localparam WORD_BITS = 16;
localparam FRAC_BITS = 8;
localparam NUM_CASES = 8;

reg                        clk;
reg                        rst_n;
reg                        start;
reg  signed [WORD_BITS-1:0] dividend_a;
reg  signed [WORD_BITS-1:0] divisor_b;
wire                        busy;
wire                        done;
wire signed [WORD_BITS-1:0] quotient;

reg signed [WORD_BITS-1:0] a_vec [0:NUM_CASES-1];
reg signed [WORD_BITS-1:0] b_vec [0:NUM_CASES-1];
reg signed [WORD_BITS-1:0] expected_vec [0:NUM_CASES-1];

integer i;
integer errors;

divider_q88 #(
    .WORD_BITS(WORD_BITS),
    .FRAC_BITS(FRAC_BITS)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .dividend_a(dividend_a),
    .divisor_b(divisor_b),
    .busy(busy),
    .done(done),
    .quotient(quotient)
);

always #5 clk = ~clk;

initial begin
    a_vec[0] = 16'sd1536;   b_vec[0] = 16'sd512;   expected_vec[0] = 16'sd768;   // 6.0 / 2.0 = 3.0
    a_vec[1] = 16'sd256;    b_vec[1] = 16'sd768;   expected_vec[1] = 16'sd85;    // 1.0 / 3.0
    a_vec[2] = -16'sd1024;  b_vec[2] = 16'sd512;   expected_vec[2] = -16'sd512;  // -4.0 / 2.0 = -2.0
    a_vec[3] = -16'sd256;   b_vec[3] = -16'sd768;  expected_vec[3] = 16'sd85;    // -1.0 / -3.0
    a_vec[4] = 16'sd128;    b_vec[4] = 16'sd128;   expected_vec[4] = 16'sd256;   // 0.5 / 0.5 = 1.0 (self)
    a_vec[5] = 16'sd32765;  b_vec[5] = 16'sd128;   expected_vec[5] = 16'sd32767; // 127.99 / 0.5
    a_vec[6] = 16'sd32767;  b_vec[6] = 16'sd4;     expected_vec[6] = 16'sd32767; // overflow -> Q88_MAX
    a_vec[7] = -16'sd32768; b_vec[7] = 16'sd4;     expected_vec[7] = -16'sd32768;// overflow -> Q88_MIN

    clk = 0;
    rst_n = 0;
    start = 0;
    dividend_a = 0;
    divisor_b = 0;
    errors = 0;

    #12 rst_n = 1;

    for (i = 0; i < NUM_CASES; i = i + 1) begin
        @(negedge clk);
        dividend_a = a_vec[i];
        divisor_b  = b_vec[i];
        start      = 1;
        @(negedge clk);
        start = 0;
        wait (done);
        if (quotient !== expected_vec[i]) begin
            errors = errors + 1;
            $display("FAIL case %0d: a=%0d b=%0d expected=%0d got=%0d",
                      i, a_vec[i], b_vec[i], expected_vec[i], quotient);
        end else begin
            $display("PASS case %0d: a=%0d b=%0d -> %0d", i, a_vec[i], b_vec[i], quotient);
        end
        @(negedge clk);
    end

    if (errors == 0)
        $display("ALL %0d DIVIDER CASES PASSED", NUM_CASES);
    else
        $display("%0d/%0d DIVIDER CASES FAILED", errors, NUM_CASES);

    $finish;
end

endmodule
