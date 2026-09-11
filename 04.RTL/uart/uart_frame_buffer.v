/*
 * Module: uart_frame_buffer
 * Packs UART bytes into 64-sample channel blocks for the ML pipeline.
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
    output reg signed [N*DATA_WIDTH-1:0]    sample_block_o,
    output reg [1:0]                        channel_o,
    output reg                              overflow_o
);

localparam CHANNEL_COUNT = 4;
localparam SAMPLE_COUNT_WIDTH = 6;

reg signed [N*DATA_WIDTH-1:0] sample_bank [0:1];
reg [1:0]                     channel_bank [0:1];
reg [1:0]                     bank_valid;
reg [1:0]                     pending_count;
reg                           read_bank;
reg                           write_bank;
reg [1:0]                     channel_index;
reg [SAMPLE_COUNT_WIDTH-1:0]  sample_index;
reg [7:0]                     msb_byte;
reg                           have_msb;

wire [DATA_WIDTH-1:0] sample_word;
wire                  accept_frame;
wire                  last_sample;
wire                  can_accept_byte;

assign sample_word  = {msb_byte, byte_i};
assign accept_frame = frame_valid_o && frame_ready_i;
assign last_sample  = (sample_index == (N - 1));
assign can_accept_byte = (pending_count < 2) || accept_frame;

always @(*) begin
    frame_valid_o  = (pending_count != 0);
    sample_block_o = sample_bank[read_bank];
    channel_o      = channel_bank[read_bank];
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sample_bank[0] <= {N*DATA_WIDTH{1'b0}};
        sample_bank[1] <= {N*DATA_WIDTH{1'b0}};
        channel_bank[0] <= 2'd0;
        channel_bank[1] <= 2'd0;
        bank_valid      <= 2'b00;
        pending_count   <= 2'd0;
        read_bank       <= 1'b0;
        write_bank      <= 1'b0;
        channel_index   <= 2'd0;
        sample_index    <= {SAMPLE_COUNT_WIDTH{1'b0}};
        msb_byte        <= 8'd0;
        have_msb        <= 1'b0;
        overflow_o      <= 1'b0;
    end else begin
        if (accept_frame) begin
            bank_valid[read_bank] <= 1'b0;
            pending_count          <= pending_count - 2'd1;
            read_bank              <= ~read_bank;
            overflow_o            <= 1'b0;
        end

        if (byte_valid_i) begin
            if (!can_accept_byte) begin
                overflow_o <= 1'b1;
            end else if (!have_msb) begin
                msb_byte <= byte_i;
                have_msb <= 1'b1;
            end else begin
                have_msb <= 1'b0;
                sample_bank[write_bank][(sample_index+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= sample_word;

                if (last_sample) begin
                    bank_valid[write_bank]   <= 1'b1;
                    channel_bank[write_bank] <= channel_index;
                    if (pending_count == 0) begin
                        read_bank <= write_bank;
                    end else if (accept_frame && (pending_count == 1)) begin
                        read_bank <= write_bank;
                    end
                    if (accept_frame) begin
                        pending_count <= pending_count;
                    end else begin
                        pending_count <= pending_count + 2'd1;
                    end
                    sample_index             <= {SAMPLE_COUNT_WIDTH{1'b0}};
                    channel_index            <= (channel_index == (CHANNEL_COUNT - 1)) ? 2'd0 :
                                                (channel_index + 2'd1);
                    write_bank               <= ~write_bank;
                end else begin
                    sample_index <= sample_index + {{(SAMPLE_COUNT_WIDTH-1){1'b0}}, 1'b1};
                end
            end
        end
    end
end

endmodule
