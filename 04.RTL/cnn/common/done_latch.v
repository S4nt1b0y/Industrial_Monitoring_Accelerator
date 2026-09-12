/*
 * Module: done_latch
 * Holds a one-cycle done pulse until it is explicitly cleared.
 *
 * Blocks in this design signal completion with a single-cycle pulse,
 * which is fine for a consumer waiting on one of them. Waiting on two
 * that finish at different times needs each pulse remembered -- ANDing
 * the raw pulses would only fire if they happened to land on the same
 * cycle.
 * Clear wins over set, so re-arming while a stale pulse arrives cannot
 * leave the latch stuck.
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
