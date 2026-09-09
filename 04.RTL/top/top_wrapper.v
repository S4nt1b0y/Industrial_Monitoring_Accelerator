module top_wrapper #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    parameter CLK_FREQ_HZ = 50000000
)(
    input  wire clk,
    input  wire rst_n,
    input  wire ml_swith,
    input  wire uart_rx_i,
    output reg Led_Normal,
    output reg Led_Unbalaced,
    output reg Led_disalaighn,
    output reg Led_desgaste
);

localparam UART_BAUD_RATE = 9600;

wire [7:0] uart_data;
wire       uart_valid;
wire       frame_valid;
wire       frame_ready;
wire       frame_overflow;

wire signed [N*DATA_WIDTH-1:0] acc_x_a;
wire signed [N*DATA_WIDTH-1:0] acc_x_b;
wire signed [N*DATA_WIDTH-1:0] acc_y_a;
wire signed [N*DATA_WIDTH-1:0] acc_y_b;

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

assign ml_valid_i  = frame_valid && !ml_swith;
assign cnn_valid_i = frame_valid &&  ml_swith;
assign frame_ready = ml_swith ? cnn_ready_o : ml_ready_o;

assign selected_valid = ml_swith ? cnn_valid_o : ml_valid_o;
assign selected_class = ml_swith ? cnn_class_o : ml_class_o;

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
    .acc_x_a_o(acc_x_a),
    .acc_x_b_o(acc_x_b),
    .acc_y_a_o(acc_y_a),
    .acc_y_b_o(acc_y_b),
    .overflow_o(frame_overflow)
);

ml_pipeline #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
) u_ml_pipeline (
    .clk(clk),
    .rst_n(rst_n),
    .acc_x_a_i(acc_x_a),
    .acc_x_b_i(acc_x_b),
    .acc_y_a_i(acc_y_a),
    .acc_y_b_i(acc_y_b),
    .valid_i(ml_valid_i),
    .ready_o(ml_ready_o),
    .valid_o(ml_valid_o),
    .class_o(ml_class_o)
);

// cnn #(
//     .DATA_WIDTH(DATA_WIDTH),
//     .N(N)
// ) u_cnn (
//     .clk(clk),
//     .rst_n(rst_n),
//     .acc_x_a_i(acc_x_a),
//     .acc_x_b_i(acc_x_b),
//     .acc_y_a_i(acc_y_a),
//     .acc_y_b_i(acc_y_b),
//     .valid_i(cnn_valid_i),
//     .ready_o(cnn_ready_o),
//     .valid_o(cnn_valid_o),
//     .class_o(cnn_class_o)
// );

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


endmodule
