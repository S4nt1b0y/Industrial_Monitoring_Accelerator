//======================================================
// ARQUIVO: lms_control.v
// UNIDADE DE CONTROLE - FSM DO LMS
//======================================================

module lms_control (

    input wire clk,
    input wire reset,
    input wire start,

    // Status vindo do datapath
    input wire tap_last,

    //==================================================
    // CONTROLE DO DATAPATH
    //==================================================

    output reg shift_enable,

    output reg reset_acc,
    output reg enable_acc,

    output reg reset_tap,
    output reg enable_tap,

    output reg calc_error,
    output reg calc_mu_error,

    output reg enable_weight,

    output reg [1:0] mul_sel,

    //==================================================
    // SAÍDAS
    //==================================================

    output reg valid,
    output reg busy
);

    //==================================================
    // ESTADOS
    //==================================================

    localparam IDLE      = 3'd0;
    localparam LOAD      = 3'd1;
    localparam CALC_Y    = 3'd2;
    localparam CALC_E    = 3'd3;
    localparam CALC_MUE  = 3'd4;
    localparam UPDATE    = 3'd5;
    localparam DONE      = 3'd6;

    reg [2:0] state;
    reg [2:0] next_state;

    //==================================================
    // REGISTRADOR DE ESTADO
    //==================================================

    always @(posedge clk or posedge reset) begin

        if (reset)
            state <= IDLE;
        else
            state <= next_state;

    end

    always @(*) begin

        // Valores padrão

        shift_enable  = 1'b0;

        reset_acc     = 1'b0;
        enable_acc    = 1'b0;

        reset_tap     = 1'b0;
        enable_tap    = 1'b0;

        calc_error    = 1'b0;
        calc_mu_error = 1'b0;

        enable_weight = 1'b0;

        mul_sel       = 2'b00;

        valid         = 1'b0;
        busy          = 1'b1;

        next_state    = state;

        case (state)

            IDLE: begin

                busy = 1'b0;

                if (start)
                    next_state = LOAD;

            end

            LOAD: begin

                shift_enable = 1'b1;

                reset_acc = 1'b1;

                reset_tap = 1'b1;

                next_state = CALC_Y;

            end

            CALC_Y: begin

                // multiplicador:
                // x[tap] * w[tap]

                mul_sel = 2'b00;

                enable_acc = 1'b1;

                enable_tap = 1'b1;

                if (tap_last)
                    next_state = CALC_E;

            end

            CALC_E: begin

                // e(n) = d(n) - y(n)

                calc_error = 1'b1;

                // prepara contador para UPDATE
                reset_tap = 1'b1;

                next_state = CALC_MUE;

            end

            CALC_MUE: begin

                // mu * erro
                //
                // Implementado por shift

                calc_mu_error = 1'b1;

                next_state = UPDATE;

            end

            UPDATE: begin

                // multiplicador:
                //
                // (mu * erro) * x[tap]

                mul_sel = 2'b10;

                enable_weight = 1'b1;

                enable_tap = 1'b1;

                if (tap_last)
                    next_state = DONE;

            end

            DONE: begin

                valid = 1'b1;
                busy  = 1'b0;

                next_state = IDLE;

            end

            default: begin

                next_state = IDLE;

            end

        endcase

    end

endmodule