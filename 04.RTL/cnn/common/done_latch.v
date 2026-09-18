/*
 * Module: done_latch
 * Holds a one-cycle done pulse until cleared.
 *
 * Used to wait for two blocks that finish on different cycles.
 * Clear has priority over set.
 * Status: implemented.
 */
module done_latch (
    input  wire clk,
    input  wire rst_n,

    input  wire set,
    input  wire clear,

    output reg  latched
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        latched <= 1'b0;
    else if (clear)
        latched <= 1'b0;
    else if (set)
        latched <= 1'b1;
end

endmodule
