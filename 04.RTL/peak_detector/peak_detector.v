module peak_detector (
    input wire clk,
    input wire reset,

    // Dados da FFT (streaming)
    input wire fft_valid,
    input wire [4:0] fft_bin,    // Bins 0 a 31 (bin 0 é ignorado no SCAN)
    input wire signed [15:0] fft_mag, // Magnitude em Q1.15

    // Controle
    input wire start,
    output reg done,

    // Resultados
    output reg [4:0] peak1_idx,  // Índice do maior pico
    output reg [4:0] peak2_idx,  // Índice do 2º maior
    output reg [4:0] peak3_idx   // Índice do 3º maior
);

    // Estados da FSM
    localparam IDLE = 2'b00;
    localparam SCAN = 2'b01;
    localparam DONE = 2'b10;

    reg [1:0] state, next_state;
    reg [4:0] bin_counter; // Conta bins válidos (1 a 31) → 31 bins no total

    // Registradores dos 3 maiores picos (valor e índice)
    reg signed [15:0] top1_val, top2_val, top3_val;
    reg [4:0] top1_idx_r, top2_idx_r, top3_idx_r;

    // ---------------------------------------------------------
    // FSM: Registrador de estado
    // ---------------------------------------------------------
    always @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else state <= next_state;
    end

    // ---------------------------------------------------------
    // FSM: Lógica de transição
    // ---------------------------------------------------------
    always @(*) begin
        next_state = state;
        case (state)
            IDLE: if (start) next_state = SCAN;
            SCAN: if (bin_counter == 31) next_state = DONE; // Todos os bins válidos processados
            DONE: if (~start) next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // ---------------------------------------------------------
    // SCAN: Varredura dos bins e comparação dos picos
    // ---------------------------------------------------------
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            bin_counter <= 0;
            done <= 0;
            top1_val <= -32768; // Menor valor possível em Q1.15
            top2_val <= -32768;
            top3_val <= -32768;
            top1_idx_r <= 0; top2_idx_r <= 0; top3_idx_r <= 0;
            peak1_idx <= 0; peak2_idx <= 0; peak3_idx <= 0;
        end else begin
            case (state)
                IDLE: begin
                    // Reinicia para a próxima janela
                    bin_counter <= 0;
                    done <= 0;
                    top1_val <= -32768;
                    top2_val <= -32768;
                    top3_val <= -32768;
                end

                SCAN: begin
                    if (fft_valid) begin
                        // Ignora o bin 0 (DC). Processa os bins 1 a 31 (31 bins úteis)
                        if (fft_bin >= 1 && fft_bin <= 31) begin
                            bin_counter <= bin_counter + 1;

                            // Comparação em cadeia: insere o novo valor no top 3
                            if (fft_mag > top1_val) begin
                                // Novo maior: desloca top1→top2, top2→top3
                                top3_val <= top2_val;
                                top3_idx_r <= top2_idx_r;
                                top2_val <= top1_val;
                                top2_idx_r <= top1_idx_r;
                                top1_val <= fft_mag;
                                top1_idx_r <= fft_bin;
                            end else if (fft_mag > top2_val) begin
                                // Novo segundo maior: desloca top2→top3
                                top3_val <= top2_val;
                                top3_idx_r <= top2_idx_r;
                                top2_val <= fft_mag;
                                top2_idx_r <= fft_bin;
                            end else if (fft_mag > top3_val) begin
                                // Novo terceiro maior
                                top3_val <= fft_mag;
                                top3_idx_r <= fft_bin;
                            end
                        end
                    end
                end

                DONE: begin
                    // Congela os resultados para a saída
                    peak1_idx <= top1_idx_r;
                    peak2_idx <= top2_idx_r;
                    peak3_idx <= top3_idx_r;
                    done <= 1;
                end
            endcase
        end
    end

endmodule
