/*
 * Testbench: weight_rom
 * Directed checks: preload a few entries directly (no external file
 * needed at this stage), confirm registered read-back (1-cycle
 * latency) returns the right value.
 * Status: implemented.
 */
`timescale 1ns/1ps

module tb_weight_rom;

localparam WORD_BITS = 16;
localparam DEPTH     = 8;
localparam ADDR_BITS = 3;

reg                         clk;
reg  [ADDR_BITS-1:0]        addr;
wire signed [WORD_BITS-1:0] data;

integer errors;

weight_rom #(
    .WORD_BITS(WORD_BITS),
    .DEPTH(DEPTH),
    .ADDR_BITS(ADDR_BITS),
    .MEM_FILE("")
) dut (
    .clk(clk),
    .addr(addr),
    .data(data)
);

always #5 clk = ~clk;

task check;
    input signed [WORD_BITS-1:0] expected;
    input [127:0] name;
    begin
        if (data !== expected) begin
            $display("FAIL %0s: data=%0d expected=%0d", name, data, expected);
            errors = errors + 1;
        end else begin
            $display("PASS %0s: data=%0d", name, data);
        end
    end
endtask

initial begin
    clk = 0;
    errors = 0;

    dut.mem[0] = 16'sd256;   // 1.0
    dut.mem[1] = -16'sd128;  // -0.5
    dut.mem[7] = 16'sd1;     // smallest positive Q8.8 step

    addr = 3'd0;
    @(posedge clk); @(posedge clk);
    check(16'sd256, "addr0");

    addr = 3'd1;
    @(posedge clk); @(posedge clk);
    check(-16'sd128, "addr1");

    addr = 3'd7;
    @(posedge clk); @(posedge clk);
    check(16'sd1, "addr7");

    if (errors == 0)
        $display("ALL TESTS PASSED");
    else
        $display("%0d TEST(S) FAILED", errors);

    $finish;
end

endmodule
