/*
 * Module: top
 * Top-level integration: sensor interface, sample buffers, MDC, FFT,
 * matrix_inv, LMS, ml_classifier, cnn, global control and output interface.
 * Status: not implemented yet.
 */
module ml_pipeline 
#(
    DATA_WIDTH=16
),(
    input  wire clk,
    input  wire rst_n,
    input  wire [DATA_WIDTH-1:0] [63:0] acc_x_a_i,
    input  wire [DATA_WIDTH-1:0] [63:0] acc_y_a_i,
    input  wire [DATA_WIDTH-1:0] [63:0] acc_x_b_i,
    input  wire [DATA_WIDTH-1:0] [63:0] acc_y_b_i,
    input  wire valid_i,
    output reg [1:0] class_o 
);

    ftu_dif #(
        .N(N),
        .M(M),
        .W(W),
        .FRACW(FRACW),
        .BF_LAT(2)
    ) fft_x_a (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_re(in_re),
        .in_im(in_im),
        .done(done),
        .out_re(out_re),
        .out_im(out_im)
    );

    ftu_dif #(
        .N(N),
        .M(M),
        .W(W),
        .FRACW(FRACW),
        .BF_LAT(2)
    ) fft_x_b (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_re(in_re),
        .in_im(in_im),
        .done(done),
        .out_re(out_re),
        .out_im(out_im)
    );

    ftu_dif #(
        .N(N),
        .M(M),
        .W(W),
        .FRACW(FRACW),
        .BF_LAT(2)
    ) fft_y_a (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_re(in_re),
        .in_im(in_im),
        .done(done),
        .out_re(out_re),
        .out_im(out_im)
    );

    ftu_dif #(
        .N(N),
        .M(M),
        .W(W),
        .FRACW(FRACW),
        .BF_LAT(2)
    ) fft_y_b (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_re(in_re),
        .in_im(in_im),
        .done(done),
        .out_re(out_re),
        .out_im(out_im)
    );

    //OBTENÇÂO DOS PICOS da FFT

    mdc_tres_picos #(
        .WIDTH(WIDTH),
        .N_FFT(N_FFT)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .pico_in(pico_in),
        .pico_valid(pico_valid),
        .pico_ready(pico_ready),
        .fs_hz(fs_hz),
        .min_k(min_k),
        .k0(k0),
        .f0(f0),
        .busy(busy),
        .done(done),
        .result_valid(result_valid)
    );

    reg [(DATA_WIDTH)*(NUM_FEATURES)-1:0] features;

    always @(posedge clk) begin //rst assincrono é uma desgraça para FPGA 
        if(!rst_n) begin 
            features <= '0;
        end else begin 
            features <= [i*(64):0 ]
        end
    end

    ml_classifier #(
        .NUM_FEATURES(NUM_FEATURES)
    )ml_i(
        .clk(clk),
        .rst_n(rst_n),
        .valid_i(valid_i),
        .features_i(features_i),
        .valid_o(valid_o),
        .class_o(class_o)
    );

    class_o = swith ? cnn : ml;
    
endmodule
