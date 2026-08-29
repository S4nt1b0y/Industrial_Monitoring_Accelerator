module mdc_tres_picos #(
    parameter WIDTH = 6,
    parameter N_FFT = 64
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,

    
    // Interface com o detector de picos


    // Um pico por vez
    input  wire [WIDTH-1:0]     pico_in,

    // Detector informa que pico_in contém um valor válido
    input  wire                 pico_valid,

    // MDC informa que está pronto para receber um pico
    output reg                  pico_ready,

    // Frequência de amostragem utilizada no cálculo de f0
    input  wire [31:0]          fs_hz,

    // Valor mínimo aceitável para k
    input  wire [WIDTH-1:0]     min_k,

   
    // Saídas
    

    // Bin fundamental
    output reg [WIDTH-1:0]      k0,

    // Frequência fundamental
    output reg [31:0]           f0,

    // Sinais de controle
    output reg                  busy,
    output reg                  done,
    output reg                  result_valid
);


    // Registradores para armazenar os 3 picos recebidos

    reg [WIDTH-1:0] pico1_reg;
    reg [WIDTH-1:0] pico2_reg;
    reg [WIDTH-1:0] pico3_reg;

    // Conta quantos picos já foram recebidos
    reg [1:0] pico_count;

    // Registradores do algoritmo de Euclides
   

    reg [WIDTH-1:0] a;
    reg [WIDTH-1:0] b;

    // Resultado intermediário:
    // m = MDC(pico1, pico2)
    reg [WIDTH-1:0] m;

    // Resultado final:
    // k = MDC(m, pico3)
    reg [WIDTH-1:0] k;


 
    // Estados da FSM
  

    localparam IDLE       = 4'd0;
    localparam RECEIVE    = 4'd1;
    localparam CHECK      = 4'd2;
    localparam LOAD_12    = 4'd3;
    localparam MDC_12     = 4'd4;
    localparam LOAD_3     = 4'd5;
    localparam MDC_3      = 4'd6;
    localparam VALIDATE   = 4'd7;
    localparam DONE_OK    = 4'd8;
    localparam DONE_ERROR = 4'd9;

    reg [3:0] state;


  
    // Multiplicação usada para calcular:
    //
    // f0 = k0 * fs / N
    //
    // WIDTH + 32 bits evitam perda no produto
   

    wire [WIDTH+31:0] freq_mult;

    assign freq_mult = k * fs_hz;



    // Máquina de Estados
   
    always @(posedge clk or posedge reset) begin

        if (reset) begin

            pico1_reg    <= 0;
            pico2_reg    <= 0;
            pico3_reg    <= 0;
            pico_count   <= 0;

            a            <= 0;
            b            <= 0;
            m            <= 0;
            k            <= 0;

            k0           <= 0;
            f0           <= 0;

            pico_ready   <= 0;
            busy         <= 0;
            done         <= 0;
            result_valid <= 0;

            state        <= IDLE;

        end

        else begin

            // done e result_valid são pulsos de um clock
            done         <= 0;
            result_valid <= 0;


            case (state)

                
                // Aguarda início de uma nova janela
               
                IDLE: begin

                    busy       <= 0;
                    pico_ready <= 0;

                    if (start) begin

                        pico_count <= 0;

                        busy       <= 1;
                        pico_ready <= 1;

                        state      <= RECEIVE;

                    end

                end


                // Recebe os 3 picos sequencialmente
                //
                // Transferência acontece quando:
                //
                // pico_valid = 1
                // pico_ready = 1
                //
                

                RECEIVE: begin

                    busy       <= 1;
                    pico_ready <= 1;


                    if (pico_valid && pico_ready) begin

                        case (pico_count)

                            // Primeiro pico
                            2'd0: begin

                                pico1_reg  <= pico_in;
                                pico_count <= 2'd1;

                            end


                            // Segundo pico
                            2'd1: begin

                                pico2_reg  <= pico_in;
                                pico_count <= 2'd2;

                            end


                            // Terceiro pico
                            2'd2: begin

                                pico3_reg  <= pico_in;

                                pico_ready <= 0;

                                state <= CHECK;

                            end


                            default: begin

                                pico_count <= 0;
                                state      <= DONE_ERROR;

                            end

                        endcase

                    end

                end


                
                // Verifica se algum pico recebido é zero
                //
                // Se:
                //
                // pico1 == 0
                // pico2 == 0
                // pico3 == 0
                //
                // a janela é considerada inválida
    
                CHECK: begin

                    pico_ready <= 0;

                    if ((pico1_reg == 0) ||
                        (pico2_reg == 0) ||
                        (pico3_reg == 0)) begin

                        k0 <= 0;
                        f0 <= 0;

                        state <= DONE_ERROR;

                    end

                    else begin

                        state <= LOAD_12;

                    end

                end


               
                // Carrega os dois primeiros picos
                //
                // m = MDC(pico1, pico2)
              
                LOAD_12: begin

                    a <= pico1_reg;
                    b <= pico2_reg;

                    state <= MDC_12;

                end


               
                // Algoritmo de Euclides por subtrações
                //
                // enquanto a != b:
                //
                // se a > b:
                //     a = a - b
                //
                // senão:
                //     b = b - a
                //
               

                MDC_12: begin

                    if (a == b) begin

                        m <= a;

                        state <= LOAD_3;

                    end

                    else if (a > b) begin

                        a <= a - b;

                    end

                    else begin

                        b <= b - a;

                    end

                end


                
                // Segundo MDC
                //
                // k = MDC(m, pico3)

                LOAD_3: begin

                    a <= m;
                    b <= pico3_reg;

                    state <= MDC_3;

                end


                
                // Calcula MDC entre m e o terceiro pico
               
                MDC_3: begin

                    if (a == b) begin

                        k <= a;

                        state <= VALIDATE;

                    end

                    else if (a > b) begin

                        a <= a - b;

                    end

                    else begin

                        b <= b - a;

                    end

                end


                
                // Validação do resultado
                //
                // Se:
                //
                // k < min_k
                //
                // resultado é inválido
                

                VALIDATE: begin

                    if (k < min_k) begin

                        k0 <= 0;
                        f0 <= 0;

                        state <= DONE_ERROR;

                    end

                    else begin

                        // Bin fundamental
                        k0 <= k;

                        // Frequência fundamental:
                        //
                        // f0 = k0 * fs / N
                        //
                        f0 <= freq_mult / N_FFT;

                        state <= DONE_OK;

                    end

                end


               
                // Finalização com resultado válido
               

                DONE_OK: begin

                    busy         <= 0;
                    pico_ready   <= 0;

                    done         <= 1;
                    result_valid <= 1;

                    state <= IDLE;

                end


                
                // Finalização com resultado inválido
                
                DONE_ERROR: begin

                    busy         <= 0;
                    pico_ready   <= 0;

                    done         <= 1;
                    result_valid <= 0;

                    state <= IDLE;

                end


                
                // Estado de segurança
               

                default: begin

                    busy         <= 0;
                    pico_ready   <= 0;
                    done         <= 0;
                    result_valid <= 0;

                    state <= IDLE;

                end

            endcase

        end

    end

endmodule
