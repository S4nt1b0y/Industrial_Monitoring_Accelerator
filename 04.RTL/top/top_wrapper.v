module top_wrapper #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    parameter CLK_FREQ_HZ = 50000000,
    // Serial baud rate. The default 9600 is used by tb_top_wrapper;
    // 08.Quartus sets 115200 for the board.
    parameter UART_BAUD_RATE = 9600,
    // Directory of the CNN weight .hex files. $readmemh resolves it
    // against the tool's working directory. The default is for
    // 08.Quartus; testbenches override it. A wrong path gives no error:
    // the ROMs load as zeros.
    parameter CNN_WEIGHTS_DIR = "../04.RTL/cnn/weights/"
)(
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [9:0]  SW,

    // GPIO para RX do CP2102
    input  wire        GPIO_0_0,

    output wire [9:0]  LEDR
);

//------------------------------------------------------------------
// Mapeamento dos sinais da placa
//------------------------------------------------------------------
wire clk;
wire rst_n;
wire ml_switch;
wire uart_rx_i;

assign clk       = CLOCK_50;
assign rst_n     = KEY[0];      // KEY é ativo em nível baixo
assign ml_switch = SW[0];
assign uart_rx_i = GPIO_0_0;

//------------------------------------------------------------------
// Sinais internos dos LEDs
//------------------------------------------------------------------
reg Led_Normal;
reg Led_Unbalaced;
reg Led_disalaighn;
reg Led_desgaste;

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

assign ml_valid_i  = frame_valid && !ml_switch;
assign cnn_valid_i = frame_valid &&  ml_switch;
assign frame_ready = ml_switch ? cnn_ready_o : ml_ready_o;

assign selected_valid = ml_switch ? cnn_valid_o : ml_valid_o;
assign selected_class = ml_switch ? cnn_class_o : ml_class_o;

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

// CNN path. Takes the same sample_block and channel as u_ml_pipeline.
// ready_o stays high for channels 2 and 3, which are discarded.
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
