//======================================================
// ARQUIVO: lms_top.v
// TOP LEVEL DO LMS
//======================================================

module lms_top #(
    parameter DATA_W   = 16,
    parameter TAPS     = 8,
    parameter FRAC_W   = 15,
    parameter MU_SHIFT = 6
)(
    input wire clk,
    input wire reset,

    input wire start,

    // Entrada x(n)
    input wire signed [DATA_W-1:0] x_in,

    // Sinal desejado d(n)
    input wire signed [DATA_W-1:0] d_in,

    // Saídas
    output wire signed [DATA_W-1:0] e_out,
    output wire signed [DATA_W-1:0] y_out,

    output wire valid,
    output wire busy,
    output wire ready     
);

    //==================================================
    // CONTROLE
    //==================================================

    wire shift_enable;

    wire reset_acc;
    wire enable_acc;

    wire reset_tap;
    wire enable_tap;

    wire calc_error;
    wire calc_mu_error;

    wire enable_weight;

    wire [1:0] mul_sel;

    //==================================================
    // STATUS
    //==================================================

    wire tap_last;

    //==================================================
    // UNIDADE DE CONTROLE
    //==================================================

    lms_control u_control (

        .clk(clk),
        .reset(reset),
        .start(start),

        .tap_last(tap_last),

        .shift_enable(shift_enable),

        .reset_acc(reset_acc),
        .enable_acc(enable_acc),

        .reset_tap(reset_tap),
        .enable_tap(enable_tap),

        .calc_error(calc_error),
        .calc_mu_error(calc_mu_error),

        .enable_weight(enable_weight),

        .mul_sel(mul_sel),

        .valid(valid),
        .busy(busy)
    );

    //==================================================
    // DATAPATH
    //==================================================

    lms_datapath #(

        .DATA_W(DATA_W),
        .TAPS(TAPS),
        .FRAC_W(FRAC_W),
        .MU_SHIFT(MU_SHIFT)

    ) u_datapath (

        .clk(clk),
        .reset(reset),

        .x_in(x_in),
        .d_in(d_in),

        .shift_enable(shift_enable),

        .reset_acc(reset_acc),
        .enable_acc(enable_acc),

        .reset_tap(reset_tap),
        .enable_tap(enable_tap),

        .calc_error(calc_error),
        .calc_mu_error(calc_mu_error),

        .enable_weight(enable_weight),

        .mul_sel(mul_sel),

        .tap_last(tap_last),

        .y_out(y_out),
        .e_out(e_out)
    );

    //==================================================
    // SINAL READY (handshaking)
    // ready = 1 quando o módulo não está ocupado
    //==================================================
    assign ready = ~busy;

endmodule