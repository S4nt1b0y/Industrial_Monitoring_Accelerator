/*
 * Module: mac
 * Single multiply-accumulate lane, pipelined to map cleanly onto one
 * FPGA DSP slice (registered multiply stage, registered accumulate
 * stage -- the standard DSP48-style MACC usage: A/B in this cycle,
 * product registered next cycle, accumulator updated the cycle after
 * that). This is the only place any CNN block multiplies two numbers;
 * conv2d, dense and anything else that needs MACs goes through
 * mac_array, never a local multiplier of its own.
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
    output reg                         valid   // pulses one cycle when acc has just been updated
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
