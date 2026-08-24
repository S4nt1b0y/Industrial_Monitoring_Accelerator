/*
 * Module: divider_q88
 * Signed Q8.8 / Q8.8 -> Q8.8 division, radix-2 shift-subtract restoring
 * algorithm (sign-magnitude: divides |a|<<FRAC_BITS by |b|, reapplies the
 * sign, saturates). WORD_BITS+FRAC_BITS cycles per division.
 * Bit-exact match to 03.Reference/matrix_inv/algorithms/gauss_jordan.py
 * (q88_divide): truncating (floor) magnitude division, no rounding.
 * Caller must guarantee divisor_b is non-zero (matrix_inv_gj checks the
 * pivot singularity epsilon before starting a divide).
 */
module divider_q88 #(
    parameter WORD_BITS = 16,
    parameter FRAC_BITS = 8
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        start,
    input  wire signed [WORD_BITS-1:0] dividend_a,
    input  wire signed [WORD_BITS-1:0] divisor_b,
    output reg                         busy,
    output reg                         done,
    output reg  signed [WORD_BITS-1:0] quotient
);

localparam EXT_BITS = WORD_BITS + FRAC_BITS;

wire [WORD_BITS-1:0] abs_a = dividend_a[WORD_BITS-1] ? (~dividend_a + 1'b1) : dividend_a;
wire [WORD_BITS-1:0] abs_b = divisor_b[WORD_BITS-1]  ? (~divisor_b  + 1'b1) : divisor_b;

reg [EXT_BITS-1:0] dividend_sh;
reg [EXT_BITS:0]   rem;
reg [EXT_BITS-1:0] divisor_ext;
reg [EXT_BITS-1:0] quot;
reg [5:0]          count;
reg                sign_result;
reg [EXT_BITS:0]   rem_shifted;

function [WORD_BITS-1:0] saturate_q88;
    input [EXT_BITS-1:0] mag;
    input                sign_bit;
    reg   [WORD_BITS-1:0] pos_max;
    reg   [WORD_BITS-1:0] neg_max_mag;
    begin
        pos_max     = {1'b0, {(WORD_BITS-1){1'b1}}};
        neg_max_mag = {1'b1, {(WORD_BITS-1){1'b0}}};
        if (sign_bit) begin
            if (mag >= neg_max_mag)
                saturate_q88 = neg_max_mag;
            else
                saturate_q88 = (~mag[WORD_BITS-1:0] + 1'b1);
        end else begin
            if (mag > pos_max)
                saturate_q88 = pos_max;
            else
                saturate_q88 = mag[WORD_BITS-1:0];
        end
    end
endfunction

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy     <= 1'b0;
        done     <= 1'b0;
        quotient <= {WORD_BITS{1'b0}};
    end else begin
        done <= 1'b0;
        if (start && !busy) begin
            sign_result <= dividend_a[WORD_BITS-1] ^ divisor_b[WORD_BITS-1];
            dividend_sh <= {abs_a, {FRAC_BITS{1'b0}}};
            divisor_ext <= {{FRAC_BITS{1'b0}}, abs_b};
            rem         <= {(EXT_BITS+1){1'b0}};
            quot        <= {EXT_BITS{1'b0}};
            count       <= 6'd0;
            busy        <= 1'b1;
        end else if (busy) begin
            if (count == EXT_BITS) begin
                busy     <= 1'b0;
                done     <= 1'b1;
                quotient <= saturate_q88(quot, sign_result);
            end else begin
                rem_shifted = {rem[EXT_BITS-1:0], dividend_sh[EXT_BITS-1]};
                dividend_sh <= dividend_sh << 1;
                if (rem_shifted >= {1'b0, divisor_ext}) begin
                    rem  <= rem_shifted - {1'b0, divisor_ext};
                    quot <= {quot[EXT_BITS-2:0], 1'b1};
                end else begin
                    rem  <= rem_shifted;
                    quot <= {quot[EXT_BITS-2:0], 1'b0};
                end
                count <= count + 1'b1;
            end
        end
    end
end

endmodule
