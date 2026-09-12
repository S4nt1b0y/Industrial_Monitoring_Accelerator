/*
 * Module: mac_array
 * Bank of NUM_MACS independent mac.v lanes, driven in parallel -- the
 * shared, DSP-oriented math resource conv2d/dense send their
 * multiply-accumulate work to instead of multiplying locally. Each
 * lane keeps its own running sum; combining partial sums across lanes
 * into one final result is the caller's job (e.g. conv2d_datapath
 * splits its taps across lanes, then adds the lane totals together
 * once at the end), not this module's -- keeps this block reusable
 * for any tap count/reduction shape.
 * NUM_MACS is the width of the parallelism: raise it to use more DSP
 * slices and fewer cycles per accumulation, lower it to use fewer.
 * Status: implemented.
 */
module mac_array #(
    parameter WORD_BITS = 16,
    parameter ACC_BITS  = 32,
    parameter NUM_MACS  = 4
) (
    input  wire clk,
    input  wire rst_n,

    input  wire [NUM_MACS-1:0]                  en,
    input  wire [NUM_MACS-1:0]                  clear,
    input  wire signed [NUM_MACS*WORD_BITS-1:0] a,
    input  wire signed [NUM_MACS*WORD_BITS-1:0] b,

    output wire signed [NUM_MACS*ACC_BITS-1:0]  acc,
    output wire [NUM_MACS-1:0]                  valid
);

genvar i;
generate
    for (i = 0; i < NUM_MACS; i = i + 1) begin : lane
        mac #(
            .WORD_BITS(WORD_BITS),
            .ACC_BITS(ACC_BITS)
        ) u_mac (
            .clk(clk),
            .rst_n(rst_n),
            .en(en[i]),
            .clear(clear[i]),
            .a(a[(i+1)*WORD_BITS-1 -: WORD_BITS]),
            .b(b[(i+1)*WORD_BITS-1 -: WORD_BITS]),
            .acc(acc[(i+1)*ACC_BITS-1 -: ACC_BITS]),
            .valid(valid[i])
        );
    end
endgenerate

endmodule
