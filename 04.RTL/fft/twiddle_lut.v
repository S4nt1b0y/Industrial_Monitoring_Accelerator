module twiddle_lut #(
    parameter W    = 16,   // largura da palavra (Q1.15)
    parameter AW   = 5,    // largura do indice k (log2(HALF))
    parameter HALF = 32    // N/2, numero de entradas na LUT
)(
    input  wire [AW-1:0]        k,
    output reg  signed [W-1:0]  w_re,
    output reg  signed [W-1:0]  w_im
);

    always @(*) begin
        case (k)
            5'd0 : begin w_re = 16'sh7FFF; w_im = 16'sh0000; end
            5'd1 : begin w_re = 16'sh7F62; w_im = 16'shF374; end
            5'd2 : begin w_re = 16'sh7D8A; w_im = 16'shE707; end
            5'd3 : begin w_re = 16'sh7A7D; w_im = 16'shDAD8; end
            5'd4 : begin w_re = 16'sh7642; w_im = 16'shCF04; end
            5'd5 : begin w_re = 16'sh70E3; w_im = 16'shC3A9; end
            5'd6 : begin w_re = 16'sh6A6E; w_im = 16'shB8E3; end
            5'd7 : begin w_re = 16'sh62F2; w_im = 16'shAECC; end
            5'd8 : begin w_re = 16'sh5A82; w_im = 16'shA57E; end
            5'd9 : begin w_re = 16'sh5134; w_im = 16'sh9D0E; end
            5'd10: begin w_re = 16'sh471D; w_im = 16'sh9592; end
            5'd11: begin w_re = 16'sh3C57; w_im = 16'sh8F1D; end
            5'd12: begin w_re = 16'sh30FC; w_im = 16'sh89BE; end
            5'd13: begin w_re = 16'sh2528; w_im = 16'sh8583; end
            5'd14: begin w_re = 16'sh18F9; w_im = 16'sh8276; end
            5'd15: begin w_re = 16'sh0C8C; w_im = 16'sh809E; end
            5'd16: begin w_re = 16'sh0000; w_im = 16'sh8000; end
            5'd17: begin w_re = 16'shF374; w_im = 16'sh809E; end
            5'd18: begin w_re = 16'shE707; w_im = 16'sh8276; end
            5'd19: begin w_re = 16'shDAD8; w_im = 16'sh8583; end
            5'd20: begin w_re = 16'shCF04; w_im = 16'sh89BE; end
            5'd21: begin w_re = 16'shC3A9; w_im = 16'sh8F1D; end
            5'd22: begin w_re = 16'shB8E3; w_im = 16'sh9592; end
            5'd23: begin w_re = 16'shAECC; w_im = 16'sh9D0E; end
            5'd24: begin w_re = 16'shA57E; w_im = 16'shA57E; end
            5'd25: begin w_re = 16'sh9D0E; w_im = 16'shAECC; end
            5'd26: begin w_re = 16'sh9592; w_im = 16'shB8E3; end
            5'd27: begin w_re = 16'sh8F1D; w_im = 16'shC3A9; end
            5'd28: begin w_re = 16'sh89BE; w_im = 16'shCF04; end
            5'd29: begin w_re = 16'sh8583; w_im = 16'shDAD8; end
            5'd30: begin w_re = 16'sh8276; w_im = 16'shE707; end
            5'd31: begin w_re = 16'sh809E; w_im = 16'shF374; end
            default: begin w_re = 16'sh0000; w_im = 16'sh0000; end
        endcase
    end

endmodule