module mdc_tres_picos #(
    parameter WIDTH = 6
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,

    input  wire [WIDTH-1:0]     pico1,
    input  wire [WIDTH-1:0]     pico2,
    input  wire [WIDTH-1:0]     pico3,

    output reg  [WIDTH-1:0]     resultado,
    output reg                  busy,
    output reg                  done,
    output reg                  result_valid
);

    // Registradores utilizados pelo algoritmo de Euclides
   

    reg [WIDTH-1:0] a;
    reg [WIDTH-1:0] b;

    reg [WIDTH-1:0] resultado_parcial;


    // Estados da máquina de estados


    localparam IDLE    = 3'd0;
    localparam LOAD_12 = 3'd1;
    localparam MDC_12  = 3'd2;
    localparam LOAD_3  = 3'd3;
    localparam MDC_3   = 3'd4;
    localparam DONE    = 3'd5;

    reg [2:0] state;


    
    // Lógica sequencial
   

    always @(posedge clk or posedge reset) begin

        if (reset) begin

            a                 <= 0;
            b                 <= 0;
            resultado_parcial <= 0;
            resultado         <= 0;

            busy              <= 0;
            done              <= 0;
            result_valid      <= 0;

            state             <= IDLE;

        end

        else begin

            // Por padrão, done e result_valid são pulsos
            done         <= 0;
            result_valid <= 0;

            case (state)

             
                // Aguarda comando de início
               

                IDLE: begin

                    busy <= 0;

                    if (start) begin
                        busy  <= 1;
                        state <= LOAD_12;
                    end

                end


              
                // Carrega pico1 e pico2
                //
                // Equivalente a:
                //
                // resultado_parcial = mdc(pico1, pico2)
              

                LOAD_12: begin

                    // if a == 0:
                    //     return b

                    if (pico1 == 0) begin

                        resultado_parcial <= pico2;
                        state             <= LOAD_3;

                    end


                    // if b == 0:
                    //     return a

                    else if (pico2 == 0) begin

                        resultado_parcial <= pico1;
                        state             <= LOAD_3;

                    end


                    // Caso normal:
                    // carregar registradores A e B

                    else begin

                        a     <= pico1;
                        b     <= pico2;
                        state <= MDC_12;

                    end

                end


               
                // Calcula MDC(pico1, pico2)
                //
                // while a != b:
                //     if a > b:
                //         a = a - b
                //     else:
                //         b = b - a
               

                MDC_12: begin

                    // terminou o MDC
                    if (a == b) begin

                        resultado_parcial <= a;
                        state             <= LOAD_3;

                    end


                    // a > b
                    else if (a > b) begin

                        a <= a - b;

                    end


                    // b > a
                    else begin

                        b <= b - a;

                    end

                end


                // Prepara o segundo MDC
                //
                // resultado_final =
                // mdc(resultado_parcial, pico3)
               

                LOAD_3: begin

                    // MDC(0, pico3) = pico3

                    if (resultado_parcial == 0) begin

                        resultado <= pico3;
                        state     <= DONE;

                    end


                    // MDC(resultado_parcial, 0)
                    // = resultado_parcial

                    else if (pico3 == 0) begin

                        resultado <= resultado_parcial;
                        state     <= DONE;

                    end


                    // Carrega o resultado parcial e o terceiro pico

                    else begin

                        a     <= resultado_parcial;
                        b     <= pico3;
                        state <= MDC_3;

                    end

                end


                // Calcula:
                //
                // MDC(resultado_parcial, pico3)
                

                MDC_3: begin

                    if (a == b) begin

                        resultado <= a;
                        state     <= DONE;

                    end

                    else if (a > b) begin

                        a <= a - b;

                    end

                    else begin

                        b <= b - a;

                    end

                end


                // Resultado disponível
          
                DONE: begin

                    busy         <= 0;
                    done         <= 1;
                    result_valid <= 1;

                    state <= IDLE;

                end


                default: begin
                    state <= IDLE;
                end

            endcase

        end

    end

endmodule
