module top_wrapper #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    parameter CLK_FREQ_HZ = 50000000,
    parameter UART_BAUD_RATE = 9600 // Mantido para compatibilidade, mas não usado
)(
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [9:0]  SW,
    input  wire        GPIO_0_0, // Não usado no teste interno
    output wire [9:0]  LEDR
);

//------------------------------------------------------------------
// Mapeamento dos sinais da placa
//------------------------------------------------------------------
wire clk;
wire rst_n;
wire ml_switch;

assign clk       = CLOCK_50;
assign rst_n     = KEY[0];      // KEY é ativo em nível baixo
assign ml_switch = SW[0];

//------------------------------------------------------------------
// Sinais internos
//------------------------------------------------------------------
reg Led_Normal;
reg Led_Unbalaced;
reg Led_disalaighn;
reg Led_desgaste;

wire signed [N*DATA_WIDTH-1:0] sample_block;
wire [1:0] frame_channel;
wire       frame_valid;
wire       frame_ready;
wire       frame_overflow; // Não usado no gerador interno

wire ml_valid_i;
wire ml_ready_o;
wire ml_valid_o;
wire [1:0] ml_class_o;

wire cnn_valid_i;
wire cnn_ready_o;
wire cnn_valid_o;
wire [1:0] cnn_class_o;

wire selected_valid;
wire [1:0] selected_class;

// Lógica de seleção (ML ou CNN)
assign ml_valid_i  = frame_valid && !ml_switch;
assign cnn_valid_i = frame_valid &&  ml_switch;
assign frame_ready = ml_switch ? cnn_ready_o : ml_ready_o;

assign selected_valid = ml_switch ? cnn_valid_o : ml_valid_o;
assign selected_class = ml_switch ? cnn_class_o : ml_class_o;

//------------------------------------------------------------------
// INSTANCIAÇÃO DO GERADOR DE DADOS INTERNO (Substitui a UART)
//------------------------------------------------------------------
data_generator #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
) u_data_gen (
    .clk(clk),
    .rst_n(rst_n),
    .ready_i(frame_ready), // Espera o pipeline estar pronto
    .valid_o(frame_valid), // Gera o pulso de válido
    .channel_o(frame_channel),
    .sample_block_o(sample_block)
);

//------------------------------------------------------------------
// PIPELINE DE ML
//------------------------------------------------------------------
ml_pipeline #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
) u_ml_pipeline (
    .clk(clk),
    .rst_n(rst_n),
    .sample_block_i(sample_block),
    .channel_i(frame_channel),
    .valid_i(ml_valid_i),
    .ready_o(ml_ready_o),
    .valid_o(ml_valid_o),
    .class_o(ml_class_o)
);

// O módulo CNN está comentado no seu código original, então mantemos assim
//cnn #( ... ) u_cnn ( ... );

//------------------------------------------------------------------
// LÓGICA DOS LEDs
//------------------------------------------------------------------
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        Led_Normal     <= 1'b0;
        Led_Unbalaced  <= 1'b0;
        Led_disalaighn <= 1'b0;
        Led_desgaste   <= 1'b0;
    end else if (selected_valid) begin
        Led_Normal     <= (selected_class == 2'd0);
        Led_disalaighn <= (selected_class == 2'd1);
        Led_Unbalaced  <= (selected_class == 2'd2);
        Led_desgaste   <= (selected_class == 2'd3);
    end else if (frame_overflow) begin
        Led_Normal     <= 1'b0;
        Led_Unbalaced  <= 1'b0;
        Led_disalaighn <= 1'b0;
        Led_desgaste   <= 1'b0;
    end
end

assign LEDR[0] = Led_Normal;
assign LEDR[1] = Led_Unbalaced;
assign LEDR[2] = Led_disalaighn;
assign LEDR[3] = Led_desgaste;
assign LEDR[9:4] = 6'b0;

endmodule