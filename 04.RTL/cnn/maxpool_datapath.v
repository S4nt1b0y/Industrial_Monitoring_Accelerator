/*
 * Module: maxpool_datapath
 * 2x2 max pooling, stride 2, one feature map per run: IMG_H x IMG_W in,
 * (IMG_H/POOL) x (IMG_W/POOL) out. The caller sequences the 8 filters,
 * the same way it sequences them through conv2d.
 *
 * The 4 window positions stream one per cycle and the running maximum is
 * a plain signed comparator -- cheap LUT logic that has no business
 * going through the DSP-oriented MAC array. Input and output are memory
 * ports with combinational reads, matching conv2d_datapath.
 * Every state decision comes from maxpool_ctrl as a one-cycle control
 * pulse; this module never decides what to do next on its own.
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
    end else begin
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

        /* Signed compare: post-ReLU maps are non-negative, but this block
         * is also useful on signed maps and an unsigned compare would pick
         * the most negative value as the maximum. */
        if (tap_stream) begin
            if (is_first_tap || (in_data > best))
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
