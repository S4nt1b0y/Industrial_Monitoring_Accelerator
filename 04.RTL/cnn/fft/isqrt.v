/*
 * Module: isqrt
 * Integer square root, restoring method: floor(sqrt(value)) of a
 * 2*WORD_BITS input in WORD_BITS iterations, one result bit per cycle.
 *
 * Uses only shifts, compares and subtracts; no multiplier.
 * The result is exact, so the magnitude matches np.abs() of the
 * reference model.
 * A Q2.30 input gives a Q1.15 result, since sqrt(x^2 * 2^30) = x * 2^15.
 * Status: implemented.
 */
module isqrt #(
    parameter WORD_BITS = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                   start,
    input  wire [2*WORD_BITS-1:0] value,

    output reg                    busy,
    output reg                    done,
    output reg  [WORD_BITS-1:0]   root
);

localparam REM_BITS = WORD_BITS + 4;

reg [2*WORD_BITS-1:0] shifter;
reg [REM_BITS-1:0]    rem;
reg [WORD_BITS-1:0]   root_acc;
reg [WORD_BITS:0]     iter;

/* one iteration: bring down the next 2 bits, compare against
 * (root_acc << 2) | 1, subtract when it fits and append that bit */
wire [REM_BITS-1:0] rem_shifted = {rem[REM_BITS-3:0], shifter[2*WORD_BITS-1 -: 2]};
wire [REM_BITS-1:0] test        = {{(REM_BITS-WORD_BITS-2){1'b0}}, root_acc, 2'b01};
wire                fits        = (rem_shifted >= test);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy     <= 1'b0;
        done     <= 1'b0;
        root     <= {WORD_BITS{1'b0}};
        shifter  <= {(2*WORD_BITS){1'b0}};
        rem      <= {REM_BITS{1'b0}};
        root_acc <= {WORD_BITS{1'b0}};
        iter     <= {(WORD_BITS+1){1'b0}};
    end else begin
        done <= 1'b0;

        if (start && !busy) begin
            shifter  <= value;
            rem      <= {REM_BITS{1'b0}};
            root_acc <= {WORD_BITS{1'b0}};
            iter     <= WORD_BITS;
            busy     <= 1'b1;
        end else if (busy) begin
            rem      <= fits ? (rem_shifted - test) : rem_shifted;
            root_acc <= {root_acc[WORD_BITS-2:0], fits};
            shifter  <= {shifter[2*WORD_BITS-3:0], 2'b00};
            iter     <= iter - 1;

            if (iter == 1) begin
                busy <= 1'b0;
                done <= 1'b1;
                root <= {root_acc[WORD_BITS-2:0], fits};
            end
        end
    end
end

endmodule
