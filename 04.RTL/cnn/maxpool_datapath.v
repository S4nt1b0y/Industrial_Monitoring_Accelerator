/*
 * Module: maxpool_datapath
 * 2x2 max pooling, stride 2, one feature map per run: IMG_H x IMG_W in,
 * (IMG_H/POOL) x (IMG_W/POOL) out. The caller runs it once per filter.
 *
 * The window positions go to a signed comparator one per cycle.
 * Input and output are memory ports; input reads have one cycle of
 * latency.
 * Control pulses come from maxpool_ctrl.
 * Status: implemented.
 */
module maxpool_datapath #(
    parameter WORD_BITS = 16,
    parameter POOL      = 2,
    parameter IMG_H     = 4,
    parameter IMG_W     = 4,
    parameter ADDR_BITS = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire pixel_init,
    input  wire tap_init,
    input  wire tap_stream,
    input  wire do_write,
    input  wire pixel_step,

    output wire tap_last,
    output wire pixel_last,

    output wire [ADDR_BITS-1:0]        in_addr,
    input  wire signed [WORD_BITS-1:0] in_data,

    output wire [ADDR_BITS-1:0]        out_addr,
    output wire signed [WORD_BITS-1:0] out_word,
    output wire                        out_we
);

localparam OUT_H = IMG_H / POOL;
localparam OUT_W = IMG_W / POOL;

integer oy, ox, ky, kx;

reg signed [WORD_BITS-1:0] best;

/* The input memory answers one cycle after the address. These registers
 * delay the tap flags by one cycle to align them with the data. */
reg stream_r, first_r;

wire is_first_tap = (ky == 0) && (kx == 0);

assign tap_last   = (kx == POOL-1) && (ky == POOL-1);
assign pixel_last = (oy == OUT_H-1) && (ox == OUT_W-1);

/* Source pixel of this tap. Stride equals the window size, so the
 * windows tile the map exactly and no read can fall out of bounds. */
assign in_addr = (oy * POOL + ky) * IMG_W + (ox * POOL + kx);

assign out_addr = oy * OUT_W + ox;
assign out_word = best;
assign out_we   = do_write;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        oy <= 0; ox <= 0; ky <= 0; kx <= 0;
        best <= {WORD_BITS{1'b0}};
        stream_r <= 1'b0;
        first_r  <= 1'b0;
    end else begin
        stream_r <= tap_stream;
        first_r  <= tap_stream && is_first_tap;

        if (pixel_init) begin
            oy <= 0;
            ox <= 0;
        end

        if (tap_init) begin
            ky <= 0;
            kx <= 0;
        end else if (tap_stream && !tap_last) begin
            if (kx == POOL-1) begin
                kx <= 0;
                ky <= ky + 1;
            end else begin
                kx <= kx + 1;
            end
        end

        /* Signed compare, so the block also works on maps with negative
         * values. Gated on the delayed flags: `in_data` belongs to the tap
         * addressed in the previous cycle. */
        if (stream_r) begin
            if (first_r || (in_data > best))
                best <= in_data;
        end

        if (pixel_step) begin
            if (ox == OUT_W-1) begin
                ox <= 0;
                oy <= oy + 1;
            end else begin
                ox <= ox + 1;
            end
        end
    end
end

endmodule
