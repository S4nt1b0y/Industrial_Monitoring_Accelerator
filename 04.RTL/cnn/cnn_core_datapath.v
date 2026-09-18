/*
 * Module: cnn_core_datapath
 * Counters and addresses for cnn_core: current input channel for
 * normalize, current filter for conv2d, and the offset of that filter's
 * pooled map in the flattened activation memory.
 *
 * Flatten order read by the dense layer: filter*POOL_ELEMS + y*POOL_W
 * + x (C order, as numpy reshape(-1)). maxpool produces y*POOL_W + x;
 * this module adds filter*POOL_ELEMS.
 * Control pulses come from cnn_core_ctrl.
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
