/*
 * Module: mac
 * One multiply-accumulate lane, pipelined to map onto one DSP block.
 *
 * Latency: operands in cycle 0, product registered in cycle 1,
 * accumulator updated in cycle 2.
 * Every multiply in the CNN path goes through mac_array.
 * Status: implemented.
 */
module mac #(
    parameter WORD_BITS = 16,  // Q8.8
    parameter ACC_BITS  = 32   // Q16.16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        en,     // a/b valid this cycle, run one MAC step
    input  wire                        clear,  // this step starts a new sum (acc <- a*b instead of acc + a*b)
    input  wire signed [WORD_BITS-1:0] a,
    input  wire signed [WORD_BITS-1:0] b,

    output reg  signed [ACC_BITS-1:0]  acc,
    output reg                         valid   // high for one cycle after acc is updated
);

reg signed [ACC_BITS-1:0] product_r;
reg                       clear_r, en_r;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        product_r <= {ACC_BITS{1'b0}};
        clear_r   <= 1'b0;
        en_r      <= 1'b0;
        acc       <= {ACC_BITS{1'b0}};
        valid     <= 1'b0;
    end else begin
        // stage 1: multiply
        product_r <= a * b;
        clear_r   <= clear;
        en_r      <= en;

        // stage 2: accumulate
        if (en_r)
            acc <= clear_r ? product_r : (acc + product_r);
        valid <= en_r;
    end
end

endmodule
