/*
 * Module: uart_rx
 * 8N1 UART receiver with fixed baud rate.
 */
module uart_rx #(
    parameter CLK_FREQ_HZ = 50000000,
    parameter BAUD_RATE   = 9600
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx_i,
    input  wire       enable_i,
    output reg        valid_o,
    output reg  [7:0] data_o
);

localparam [1:0] S_IDLE  = 2'd0;
localparam [1:0] S_START = 2'd1;
localparam [1:0] S_DATA  = 2'd2;
localparam [1:0] S_STOP  = 2'd3;
localparam CLOCKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE;

reg [1:0]  state;
reg [1:0]  next_state;
reg [31:0] sample_count;
reg [2:0]  bit_index;
reg [7:0]  shift_reg;

always @(*) begin
    next_state = state;

    if (!enable_i) begin
        next_state = S_IDLE;
    end else begin
        case (state)
            S_IDLE: begin
                if (!rx_i) begin
                    next_state = S_START;
                end
            end

            S_START: begin
                if (sample_count == ((CLOCKS_PER_BIT - 1) >> 1)) begin
                    if (!rx_i) begin
                        next_state = S_DATA;
                    end else begin
                        next_state = S_IDLE;
                    end
                end
            end

            S_DATA: begin
                if ((sample_count == (CLOCKS_PER_BIT - 1)) &&
                    (bit_index == 3'd7)) begin
                    next_state = S_STOP;
                end
            end

            S_STOP: begin
                if (sample_count == (CLOCKS_PER_BIT - 1)) begin
                    next_state = S_IDLE;
                end
            end

            default: begin
                next_state = S_IDLE;
            end
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state        <= S_IDLE;
        sample_count <= 32'd0;
        bit_index    <= 3'd0;
        shift_reg    <= 8'd0;
        data_o       <= 8'd0;
        valid_o      <= 1'b0;
    end else begin
        state   <= next_state;
        valid_o <= 1'b0;

        if (!enable_i) begin
            sample_count <= 32'd0;
            bit_index    <= 3'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    sample_count <= 32'd0;
                    bit_index    <= 3'd0;
                end

                S_START: begin
                    if (sample_count == ((CLOCKS_PER_BIT - 1) >> 1)) begin
                        sample_count <= 32'd0;
                    end else begin
                        sample_count <= sample_count + 32'd1;
                    end
                end

                S_DATA: begin
                    if (sample_count == (CLOCKS_PER_BIT - 1)) begin
                        sample_count <= 32'd0;
                        shift_reg[bit_index] <= rx_i;
                        if (bit_index == 3'd7) begin
                            bit_index <= 3'd0;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        sample_count <= sample_count + 32'd1;
                    end
                end

                S_STOP: begin
                    if (sample_count == (CLOCKS_PER_BIT - 1)) begin
                        sample_count <= 32'd0;
                        if (rx_i) begin
                            data_o  <= shift_reg;
                            valid_o <= 1'b1;
                        end
                    end else begin
                        sample_count <= sample_count + 32'd1;
                    end
                end

                default: begin
                    sample_count <= 32'd0;
                    bit_index    <= 3'd0;
                end
            endcase
        end
    end
end

endmodule
