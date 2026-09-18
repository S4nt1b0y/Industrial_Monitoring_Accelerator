/*
 * Module: fir_coef_rom
 * 32 Q1.15 coefficients of the anti-alias FIR of each /8 stage:
 * firwin(32, 1/8, hamming).
 *
 * Symmetric (linear phase). The taps sum to 32768: unity DC gain.
 * Combinational read: 32 constants, mapped to logic.
 * Status: implemented.
 */
module fir_coef_rom #(
    parameter WORD_BITS = 16
) (
    input  wire [4:0]                  addr,
    output reg  signed [WORD_BITS-1:0] coef
);

always @* begin
    case (addr)
        5'd0:  coef = -16'sd10;
        5'd1:  coef = -16'sd36;
        5'd2:  coef = -16'sd75;
        5'd3:  coef = -16'sd132;
        5'd4:  coef = -16'sd198;
        5'd5:  coef = -16'sd244;
        5'd6:  coef = -16'sd231;
        5'd7:  coef = -16'sd112;
        5'd8:  coef =  16'sd152;
        5'd9:  coef =  16'sd582;
        5'd10: coef =  16'sd1167;
        5'd11: coef =  16'sd1861;
        5'd12: coef =  16'sd2589;
        5'd13: coef =  16'sd3257;
        5'd14: coef =  16'sd3768;
        5'd15: coef =  16'sd4046;
        5'd16: coef =  16'sd4046;
        5'd17: coef =  16'sd3768;
        5'd18: coef =  16'sd3257;
        5'd19: coef =  16'sd2589;
        5'd20: coef =  16'sd1861;
        5'd21: coef =  16'sd1167;
        5'd22: coef =  16'sd582;
        5'd23: coef =  16'sd152;
        5'd24: coef = -16'sd112;
        5'd25: coef = -16'sd231;
        5'd26: coef = -16'sd244;
        5'd27: coef = -16'sd198;
        5'd28: coef = -16'sd132;
        5'd29: coef = -16'sd75;
        5'd30: coef = -16'sd36;
        5'd31: coef = -16'sd10;
        default: coef = 16'sd0;
    endcase
end

endmodule
