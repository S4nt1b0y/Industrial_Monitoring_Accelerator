/*
 * Module: twiddle_rom
 * Rotation factors for the 64-point radix-2 DIF FFT: 32 (cos, sin)
 * pairs in Q1.15, W_k = exp(-j*2*pi*k/64) for k = 0..31. Stored as
 * cos and positive sin; the butterfly applies the minus sign of the
 * exponent when it combines the products, so no negated copy is kept
 * here. Combinational read -- the table is small enough that a
 * registered output would only add latency to the butterfly.
 * cos(0) rounds to 32768, which does not fit Q1.15, so it saturates to
 * 32767 (1 LSB low). Explicit storage of the rotation factors is a
 * requirement of the problem statement (Secao 3.2).
 * Status: implemented.
 */
module twiddle_rom #(
    parameter WORD_BITS = 16
) (
    input  wire [4:0]                  k,
    output reg  signed [WORD_BITS-1:0] cos_out,
    output reg  signed [WORD_BITS-1:0] sin_out
);

always @* begin
    case (k)
        5'd0:  begin cos_out =  16'sd32767; sin_out =  16'sd0;     end
        5'd1:  begin cos_out =  16'sd32610; sin_out =  16'sd3212;  end
        5'd2:  begin cos_out =  16'sd32138; sin_out =  16'sd6393;  end
        5'd3:  begin cos_out =  16'sd31357; sin_out =  16'sd9512;  end
        5'd4:  begin cos_out =  16'sd30274; sin_out =  16'sd12540; end
        5'd5:  begin cos_out =  16'sd28899; sin_out =  16'sd15447; end
        5'd6:  begin cos_out =  16'sd27246; sin_out =  16'sd18205; end
        5'd7:  begin cos_out =  16'sd25330; sin_out =  16'sd20788; end
        5'd8:  begin cos_out =  16'sd23170; sin_out =  16'sd23170; end
        5'd9:  begin cos_out =  16'sd20788; sin_out =  16'sd25330; end
        5'd10: begin cos_out =  16'sd18205; sin_out =  16'sd27246; end
        5'd11: begin cos_out =  16'sd15447; sin_out =  16'sd28899; end
        5'd12: begin cos_out =  16'sd12540; sin_out =  16'sd30274; end
        5'd13: begin cos_out =  16'sd9512;  sin_out =  16'sd31357; end
        5'd14: begin cos_out =  16'sd6393;  sin_out =  16'sd32138; end
        5'd15: begin cos_out =  16'sd3212;  sin_out =  16'sd32610; end
        5'd16: begin cos_out =  16'sd0;     sin_out =  16'sd32767; end
        5'd17: begin cos_out = -16'sd3212;  sin_out =  16'sd32610; end
        5'd18: begin cos_out = -16'sd6393;  sin_out =  16'sd32138; end
        5'd19: begin cos_out = -16'sd9512;  sin_out =  16'sd31357; end
        5'd20: begin cos_out = -16'sd12540; sin_out =  16'sd30274; end
        5'd21: begin cos_out = -16'sd15447; sin_out =  16'sd28899; end
        5'd22: begin cos_out = -16'sd18205; sin_out =  16'sd27246; end
        5'd23: begin cos_out = -16'sd20788; sin_out =  16'sd25330; end
        5'd24: begin cos_out = -16'sd23170; sin_out =  16'sd23170; end
        5'd25: begin cos_out = -16'sd25330; sin_out =  16'sd20788; end
        5'd26: begin cos_out = -16'sd27246; sin_out =  16'sd18205; end
        5'd27: begin cos_out = -16'sd28899; sin_out =  16'sd15447; end
        5'd28: begin cos_out = -16'sd30274; sin_out =  16'sd12540; end
        5'd29: begin cos_out = -16'sd31357; sin_out =  16'sd9512;  end
        5'd30: begin cos_out = -16'sd32138; sin_out =  16'sd6393;  end
        5'd31: begin cos_out = -16'sd32610; sin_out =  16'sd3212;  end
        default: begin cos_out = 16'sd0;    sin_out =  16'sd0;     end
    endcase
end

endmodule
