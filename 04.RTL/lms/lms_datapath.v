//======================================================
// ARQUIVO: lms_datapath.v
//
// DATAPATH DO LMS 
//   - 8 taps, Q1.15, 1 multiplicador compartilhado
//======================================================

module lms_datapath #(
    parameter DATA_W  = 16,
    parameter TAPS    = 8,
    parameter FRAC_W  = 15,
    parameter MU_SHIFT = 6
)(
    input wire clk,
    input wire reset,
    input wire signed [DATA_W-1:0] x_in,
    input wire signed [DATA_W-1:0] d_in,

    input wire shift_enable,
    input wire reset_acc,
    input wire enable_acc,
    input wire reset_tap,
    input wire enable_tap,
    input wire calc_error,
    input wire calc_mu_error,
    input wire enable_weight,
    input wire [1:0] mul_sel,

    output wire tap_last,
    output wire signed [DATA_W-1:0] y_out,
    output wire signed [DATA_W-1:0] e_out
);

    //==================================================
    // FUNÇÃO clog2 
    //==================================================
    function integer clog2;
        input integer value;
        integer temp;
        begin
            clog2 = 0;
            temp = value - 1;
            while (temp > 0) begin
                clog2 = clog2 + 1;
                temp = temp >> 1;
            end
        end
    endfunction

    //==================================================
    // LARGURAS
    //==================================================
    localparam PROD_W   = 2 * DATA_W;
    localparam TAP_WIDTH = clog2(TAPS);   // = 3 para TAPS=8
    localparam ACC_W    = PROD_W + TAP_WIDTH;

    //==================================================
    // SATURAÇÃO PARA Q1.15
    //==================================================
    function signed [DATA_W-1:0] saturate_15;
        input signed [DATA_W-1:0] value;
        begin
            if (value > 16'sd32767)
                saturate_15 = 16'sd32767;
            else if (value < -16'sd32768)
                saturate_15 = -16'sd32768;
            else
                saturate_15 = value;
        end
    endfunction

    //==================================================
    // BANCOS DE REGISTRADORES
    //==================================================
    reg signed [DATA_W-1:0] x_mem [0:TAPS-1];
    reg signed [DATA_W-1:0] w_mem [0:TAPS-1];
    reg [TAP_WIDTH-1:0] tap;
    reg signed [ACC_W-1:0] accumulator;

    reg signed [DATA_W-1:0] desired_reg;
    reg signed [DATA_W-1:0] y_reg;
    reg signed [DATA_W-1:0] error_reg;
    reg signed [DATA_W-1:0] mu_error_reg;
    reg signed [PROD_W-1:0] mult_result;

    localparam MUL_Y  = 2'b00;
    localparam MUL_DW = 2'b10;
    integer i;

    //==================================================
    // HISTÓRICO DE AMOSTRAS
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < TAPS; i = i + 1)
                x_mem[i] <= {DATA_W{1'b0}};        
            desired_reg <= {DATA_W{1'b0}};        
        end
        else if (shift_enable) begin
            for (i = TAPS-1; i > 0; i = i - 1)
                x_mem[i] <= x_mem[i-1];
            x_mem[0] <= x_in;
            desired_reg <= d_in;
        end
    end

    //==================================================
    // CONTADOR DE TAP
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tap <= {TAP_WIDTH{1'b0}};              
        end
        else if (reset_tap) begin
            tap <= {TAP_WIDTH{1'b0}};              
        end
        else if (enable_tap) begin
            if (tap == TAPS-1)
                tap <= {TAP_WIDTH{1'b0}};          
            else
                tap <= tap + 1'b1;
        end
    end

    assign tap_last = (tap == TAPS-1);

    //==================================================
    // MULTIPLICADOR COMPARTILHADO
    //==================================================
    always @(*) begin
        case (mul_sel)
            MUL_Y:  mult_result = x_mem[tap] * w_mem[tap];
            MUL_DW: mult_result = mu_error_reg * x_mem[tap];
            default: mult_result = {PROD_W{1'b0}}; 
        endcase
    end

    //==================================================
    // ACUMULADOR
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            accumulator <= {ACC_W{1'b0}};           
        end
        else if (reset_acc) begin
            accumulator <= {ACC_W{1'b0}};          
        end
        else if (enable_acc) begin
            accumulator <= accumulator + mult_result;
        end
    end

    //==================================================
    // CÁLCULO DE Y E E 
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            y_reg     <= {DATA_W{1'b0}};            
            error_reg <= {DATA_W{1'b0}};            
        end
        else if (calc_error) begin
            y_reg <= saturate_15(accumulator >>> FRAC_W);
            error_reg <= saturate_15(desired_reg - y_reg);
        end
    end

    //==================================================
    // MU * ERRO (shift)
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            mu_error_reg <= {DATA_W{1'b0}};         
        end
        else if (calc_mu_error) begin
            mu_error_reg <= error_reg >>> MU_SHIFT;
        end
    end

    //==================================================
    // ATUALIZAÇÃO DOS PESOS 
    //==================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < TAPS; i = i + 1)
                w_mem[i] <= {DATA_W{1'b0}};         
        end
        else if (enable_weight) begin
            w_mem[tap] <= saturate_15(w_mem[tap] + (mult_result >>> FRAC_W));
        end
    end

    //==================================================
    // SAÍDAS
    //==================================================
    assign y_out = y_reg;
    assign e_out = error_reg;

endmodule