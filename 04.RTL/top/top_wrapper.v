module top_wrapper #(
    parameter DATA_WIDTH     = 16,
    parameter N              = 64,
    parameter CLK_FREQ_HZ    = 50000000,
    parameter UART_BAUD_RATE = 115200,
    parameter CNN_WEIGHTS_DIR = "../04.RTL/cnn/weights/"
)(
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [9:0]  SW,

    // GPIO para RX do CP2102
    input  wire        GPIO_0_0,

    output wire [9:0]  LEDR,
     
    // Barramentos para os displays de 7 segmentos da DE0-CV
    output wire [6:0]  HEX0,
    output wire [6:0]  HEX1,
    output wire [6:0]  HEX2
);

//------------------------------------------------------------------
// Mapeamento dos sinais da placa
//------------------------------------------------------------------
wire clk;
wire rst_n;
wire ml_switch;
wire uart_rx_i;

assign clk       = CLOCK_50;
assign rst_n     = KEY[0];      // KEY é ativo em nível baixo (0 pressionado)
assign ml_switch = SW[0];
assign uart_rx_i = GPIO_0_0;

//------------------------------------------------------------------
// Sinais internos dos LEDs e registradores
//------------------------------------------------------------------
reg Led_Normal;
reg Led_Unbalaced;
reg Led_disalaighn;
reg Led_desgaste;

reg ready_latched;

wire [7:0] uart_data;
wire       uart_valid;
wire       frame_valid;
wire       frame_ready;
wire       frame_overflow;
wire signed [N*DATA_WIDTH-1:0] sample_block;
wire [1:0] frame_channel;

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
wire active_ready;

assign ml_valid_i     = frame_valid && !ml_switch;
assign cnn_valid_i    = frame_valid &&  ml_switch;
assign frame_ready    = ml_switch ? cnn_ready_o : ml_ready_o;

assign selected_valid = ml_switch ? cnn_valid_o : ml_valid_o;
assign selected_class = ml_switch ? cnn_class_o : ml_class_o;
assign active_ready   = ml_switch ? cnn_ready_o : ml_ready_o;

uart_rx #(
    .CLK_FREQ_HZ(CLK_FREQ_HZ),
    .BAUD_RATE(UART_BAUD_RATE)
) u_uart_rx (
    .clk(clk),
    .rst_n(rst_n),
    .rx_i(uart_rx_i),
    .enable_i(1'b1),
    .valid_o(uart_valid),
    .data_o(uart_data)
);

uart_frame_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
) u_uart_frame_buffer (
    .clk(clk),
    .rst_n(rst_n),
    .byte_i(uart_data),
    .byte_valid_i(uart_valid),
    .frame_ready_i(frame_ready),
    .frame_valid_o(frame_valid),
    .sample_block_o(sample_block),
    .channel_o(frame_channel),
    .overflow_o(frame_overflow)
);

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

cnn #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N),
    .KERNEL_FILE({CNN_WEIGHTS_DIR, "conv1_kernels.hex"}),
    .CBIAS_FILE ({CNN_WEIGHTS_DIR, "conv1_bias.hex"}),
    .DBIAS_FILE ({CNN_WEIGHTS_DIR, "dense_b.hex"}),
    .DW0_FILE   ({CNN_WEIGHTS_DIR, "dense_w_c0.hex"}),
    .DW1_FILE   ({CNN_WEIGHTS_DIR, "dense_w_c1.hex"}),
    .DW2_FILE   ({CNN_WEIGHTS_DIR, "dense_w_c2.hex"}),
    .DW3_FILE   ({CNN_WEIGHTS_DIR, "dense_w_c3.hex"})
) u_cnn (
    .clk(clk),
    .rst_n(rst_n),
    .sample_block_i(sample_block),
    .channel_i(frame_channel),
    .valid_i(cnn_valid_i),
    .ready_o(cnn_ready_o),
    .valid_o(cnn_valid_o),
    .class_o(cnn_class_o)
);

// Lógica de latched e controle dos LEDs
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        Led_Normal     <= 1'b0;
        Led_Unbalaced  <= 1'b0;
        Led_disalaighn <= 1'b0;
        Led_desgaste   <= 1'b0;
        ready_latched  <= 1'b0;
    end else begin
        // Trava do ready (executado de forma independente)
        if (active_ready) begin
            ready_latched <= 1'b1;
        end

        // Atualização das classes
        if (selected_valid) begin
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
end

// Atribuições dos LEDs
assign LEDR[0] = Led_Normal;
assign LEDR[1] = Led_Unbalaced;
assign LEDR[2] = Led_disalaighn;
assign LEDR[3] = Led_desgaste;
assign LEDR[4] = frame_overflow;
assign LEDR[5] = frame_valid;
assign LEDR[8:6] = 3'b0;
assign LEDR[9] = !KEY[0]; // Acende quando o botão de reset é pressionado

//------------------------------------------------------------------
// Decodificação dos Displays de 7 Segmentos (Ativo em 0)
//------------------------------------------------------------------

// HEX0: Mostra o canal (0 a 3)
reg [6:0] hex0_reg;
always @(*) begin
    case (frame_channel)
        2'd0: hex0_reg = 7'b100_0000; // '0'
        2'd1: hex0_reg = 7'b111_1001; // '1'
        2'd2: hex0_reg = 7'b010_0100; // '2'
        2'd3: hex0_reg = 7'b011_0000; // '3'
        default: hex0_reg = 7'b111_1111; // Apagado
    endcase
end
assign HEX0 = hex0_reg;

// HEX1: Indica o modelo selecionado (ML = 'm', CNN = 'C')
assign HEX1 = ml_switch ? 7'b100_0110 : 7'b010_1010;

// HEX2: Mostra 'r' se o sinal de ready tiver ocorrido
assign HEX2 = ready_latched ? 7'b010_1111 : 7'b111_1111;

endmodule