/*
 * Module: cnn_core_datapath
 * Counters and derived addresses for cnn_core: which input channel is
 * being normalized, which filter is being convolved, and where that
 * filter's pooled map lands in the flattened activation memory.
 *
 * The flatten offset is the one number in this design that silently
 * corrupts everything if it is wrong: the dense layer reads index
 * filter*POOL_ELEMS + y*POOL_W + x (C order, `pooled.reshape(-1)` in
 * 03.Reference/cnn/reference.py). Max-pool emits y*POOL_W + x on its own,
 * so all that is missing is the filter stride, added here.
 * Every state decision comes from cnn_core_ctrl as a one-cycle control
 * pulse; this module never decides what to do next on its own.
 * Status: implemented.
 */
module cnn_core_datapath #(
    parameter N_FILTERS   = 8,
    parameter N_CHANNELS  = 2,
    parameter POOL_ELEMS  = 256,  // per filter, after 2x2 pooling of 32x32
    parameter FILT_BITS   = 3,
    parameter CHAN_BITS   = 1,
    parameter ADDR_BITS   = 12
) (
    input  wire clk,
    input  wire rst_n,

    input  wire chan_init,
    input  wire chan_step,
    input  wire filter_init,
    input  wire filter_step,

    output wire [CHAN_BITS-1:0] chan_idx,
    output wire                 chan_last,
    output wire [FILT_BITS-1:0] filter_idx,
    output wire                 filter_last,

    input  wire [ADDR_BITS-1:0] pool_out_addr,
    output wire [ADDR_BITS-1:0] flat_addr
);

reg [CHAN_BITS-1:0] chan_r;
reg [FILT_BITS-1:0] filter_r;

assign chan_idx    = chan_r;
assign chan_last   = (chan_r == N_CHANNELS-1);
assign filter_idx  = filter_r;
assign filter_last = (filter_r == N_FILTERS-1);

assign flat_addr = pool_out_addr + (filter_r * POOL_ELEMS);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        chan_r   <= {CHAN_BITS{1'b0}};
        filter_r <= {FILT_BITS{1'b0}};
    end else begin
        if (chan_init)
            chan_r <= {CHAN_BITS{1'b0}};
        else if (chan_step)
            chan_r <= chan_r + {{(CHAN_BITS-1){1'b0}}, 1'b1};

        if (filter_init)
            filter_r <= {FILT_BITS{1'b0}};
        else if (filter_step)
            filter_r <= filter_r + {{(FILT_BITS-1){1'b0}}, 1'b1};
    end
end

endmodule
