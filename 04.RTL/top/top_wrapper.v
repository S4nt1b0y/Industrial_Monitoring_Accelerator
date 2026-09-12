module top_wrapper #(
    parameter DATA_WIDTH  = 16,
    parameter N           = 64,
    parameter CLK_FREQ_HZ = 50000000,
    // Was a fixed 9600. Promoted to a parameter because the two paths
    // want different rates and the board, not the RTL, should decide:
    // the ML path classifies from a single 512-byte frame, while the CNN
    // path needs 576 of them (36,864 raw samples per channel) for one
    // result -- over five minutes at 9600, about 26 seconds at 115200.
    // Default left at 9600 so existing testbenches and bring-up notes
    // are unaffected; set 115200 when synthesising for the CNN demo.
    parameter UART_BAUD_RATE = 9600
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

// Takes sample_block/channel like u_ml_pipeline above, not the four
// separate acc_* buses the earlier commented-out sketch assumed: the
// frame buffer only ever exposes one channel block at a time. The CNN
// keeps ready_o asserted for channels 2 and 3 as well, which it
// discards, so ingestion never stalls on them.
cnn #(
    .DATA_WIDTH(DATA_WIDTH),
    .N(N)
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


endmodule
