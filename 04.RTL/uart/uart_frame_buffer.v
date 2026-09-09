/*
 * Module: uart_frame_buffer
 * Packs UART bytes into four 64-sample channels for the ML pipeline.
 */
module uart_frame_buffer #(
    parameter DATA_WIDTH = 16,
    parameter N          = 64
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire [7:0]                       byte_i,
    input  wire                             byte_valid_i,
    input  wire                             frame_ready_i,
    output reg                              frame_valid_o,
    output reg signed [N*DATA_WIDTH-1:0]    acc_x_a_o,
    output reg signed [N*DATA_WIDTH-1:0]    acc_x_b_o,
    output reg signed [N*DATA_WIDTH-1:0]    acc_y_a_o,
    output reg signed [N*DATA_WIDTH-1:0]    acc_y_b_o,
    output reg                              overflow_o
);

localparam CHANNEL_COUNT = 4;
localparam SAMPLE_COUNT_WIDTH = 6;

reg [1:0] channel_index;
reg [SAMPLE_COUNT_WIDTH-1:0] sample_index;
reg [7:0] msb_byte;
reg       have_msb;

wire [DATA_WIDTH-1:0] sample_word;
wire                  accept_frame;
wire                  last_sample;

assign sample_word  = {msb_byte, byte_i};
assign accept_frame = frame_valid_o && frame_ready_i;
assign last_sample  = (channel_index == (CHANNEL_COUNT - 1)) &&
                      (sample_index == (N - 1));

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        frame_valid_o <= 1'b0;
        acc_x_a_o     <= {N*DATA_WIDTH{1'b0}};
        acc_x_b_o     <= {N*DATA_WIDTH{1'b0}};
        acc_y_a_o     <= {N*DATA_WIDTH{1'b0}};
        acc_y_b_o     <= {N*DATA_WIDTH{1'b0}};
        overflow_o    <= 1'b0;
        channel_index <= 2'd0;
        sample_index  <= {SAMPLE_COUNT_WIDTH{1'b0}};
        msb_byte      <= 8'd0;
        have_msb      <= 1'b0;
    end else begin
        if (accept_frame) begin
            frame_valid_o <= 1'b0;
            overflow_o    <= 1'b0;
            channel_index <= 2'd0;
            sample_index  <= {SAMPLE_COUNT_WIDTH{1'b0}};
            have_msb      <= 1'b0;
        end

        if (byte_valid_i) begin
            if (frame_valid_o && !accept_frame) begin
                overflow_o <= 1'b1;
            end else if (!have_msb) begin
                msb_byte <= byte_i;
                have_msb <= 1'b1;
            end else begin
                have_msb <= 1'b0;

                case (channel_index)
                    2'd0: acc_x_a_o[(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;
                    2'd1: acc_x_b_o[(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;
                    2'd2: acc_y_a_o[(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;
                    2'd3: acc_y_b_o[(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;
                    default: begin
                        acc_x_a_o[(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;
                    end
                endcase

                if (last_sample) begin
                    frame_valid_o <= 1'b1;
                end else if (sample_index == (N - 1)) begin
                    sample_index  <= {SAMPLE_COUNT_WIDTH{1'b0}};
                    channel_index <= channel_index + 2'd1;
                end else begin
                    sample_index <= sample_index + {{(SAMPLE_COUNT_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end
end

endmodule
