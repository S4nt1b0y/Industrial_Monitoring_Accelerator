module butterfly_dif #(
    parameter W     = 16,   // largura da palavra (Q1.FRACW)
    parameter FRACW = 15    // bits fracionarios
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire  signed [W-1:0]        a_re, a_im,
    input  wire  signed [W-1:0]        b_re, b_im,
    input  wire  signed [W-1:0]        w_re, w_im,
    output reg   signed [W-1:0]        x0_re, x0_im,
    output reg   signed [W-1:0]        x1_re, x1_im
);

    // soma/diferenca
    reg signed [W:0]   sum_re_r, sum_im_r;
    reg signed [W:0]   diff_re_r, diff_im_r;
    reg signed [W-1:0] w_re_r, w_im_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_re_r  <= 0; sum_im_r  <= 0;
            diff_re_r <= 0; diff_im_r <= 0;
            w_re_r    <= 0; w_im_r    <= 0;
        end else begin
            sum_re_r  <= a_re + b_re;
            sum_im_r  <= a_im + b_im;
            diff_re_r <= a_re - b_re;
            diff_im_r <= a_im - b_im;
            w_re_r    <= w_re;
            w_im_r    <= w_im;
        end
    end

    // multiplicacao pelo twiddle
    reg signed [2*W+1:0] pr, pi;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x0_re <= 0; x0_im <= 0;
            x1_re <= 0; x1_im <= 0;
        end else begin
            // Ramo superior: (a+b)/2
            x0_re <= sum_re_r[W:1];
            x0_im <= sum_im_r[W:1];

            pr = diff_re_r * w_re_r - diff_im_r * w_im_r;
            pi = diff_re_r * w_im_r + diff_im_r * w_re_r;

            x1_re <= pr[W+FRACW : FRACW+1];
            x1_im <= pi[W+FRACW : FRACW+1];
        end
    end

endmodule