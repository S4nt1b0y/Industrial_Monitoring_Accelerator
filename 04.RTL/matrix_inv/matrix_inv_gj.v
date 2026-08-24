/*
 * Module: matrix_inv_gj
 * Gauss-Jordan elimination with partial pivoting on the augmented matrix
 * [A | I | b], N in {2,3,4}. Normalize step uses one shared iterative
 * divider_q88 (serial, one column at a time); eliminate step updates a
 * whole row's M=2N+1 columns per cycle (row-parallel MACs).
 */
module matrix_inv_gj #(
    parameter N         = 4,
    parameter WORD_BITS = 16,
    parameter FRAC_BITS = 8
) (
    input  wire                            clk,
    input  wire                            rst_n,

    input  wire                            start,
    input  wire signed [N*N*WORD_BITS-1:0] a_in,
    input  wire signed [N*WORD_BITS-1:0]   b_in,

    output reg                             busy,
    output reg                             done,
    output reg                             singular,
    output reg  signed [N*WORD_BITS-1:0]   x_out,
    output reg  signed [N*N*WORD_BITS-1:0] ainv_out
);

localparam M = 2*N + 1;
localparam signed [WORD_BITS-1:0]   ONE_Q88   = 1 <<< FRAC_BITS;
localparam signed [WORD_BITS-1:0]   Q88_MAX_C = {1'b0, {(WORD_BITS-1){1'b1}}};
localparam signed [WORD_BITS-1:0]   Q88_MIN_C = {1'b1, {(WORD_BITS-1){1'b0}}};
localparam signed [WORD_BITS-1:0]   EPSILON_Q88 = 16'sd4;

localparam S_IDLE         = 4'd0,
           S_LOAD          = 4'd1,
           S_PIVOT_SEARCH  = 4'd2,
           S_PIVOT_SWAP    = 4'd3,
           S_NORM_ISSUE    = 4'd4,
           S_NORM_WAIT     = 4'd5,
           S_ELIM_ROW      = 4'd6,
           S_NEXT_COLUMN   = 4'd7,
           S_DONE          = 4'd8,
           S_SINGULAR_DONE = 4'd9;

reg [3:0] state;

reg signed [WORD_BITS-1:0] aug [0:N-1][0:M-1];

reg [2:0] k;
reg [3:0] col_idx; // range 0..M-1 = 0..2N, wider than k/elim_idx/pivot_row (0..N-1)
reg [2:0] elim_idx;
reg [2:0] pivot_row;
reg signed [WORD_BITS-1:0] pivot_val;

integer ii, jj, srch_i;
reg [2:0] best_row_v;
reg signed [WORD_BITS-1:0] best_val_v;

/* Shared iterative divider for the normalize step. */
wire                        div_start = (state == S_NORM_ISSUE);
wire signed [WORD_BITS-1:0] div_a = aug[k][col_idx];
wire signed [WORD_BITS-1:0] div_b = pivot_val;
wire                        div_busy;
wire                        div_done;
wire signed [WORD_BITS-1:0] div_quotient;

divider_q88 #(
    .WORD_BITS(WORD_BITS),
    .FRAC_BITS(FRAC_BITS)
) u_divider (
    .clk(clk),
    .rst_n(rst_n),
    .start(div_start),
    .dividend_a(div_a),
    .divisor_b(div_b),
    .busy(div_busy),
    .done(div_done),
    .quotient(div_quotient)
);

function signed [WORD_BITS-1:0] abs_q88;
    input signed [WORD_BITS-1:0] v;
    begin
        abs_q88 = v[WORD_BITS-1] ? -v : v;
    end
endfunction

/* row_val - round(factor * pivot_row_val), each stage saturated to Q8.8,
 * matching fixed_point.rescale_acc_to_q88() + the outer np.clip() in
 * gauss_jordan.solve_fixed(). */
function signed [WORD_BITS-1:0] eliminate_col;
    input signed [WORD_BITS-1:0] row_val;
    input signed [WORD_BITS-1:0] factor;
    input signed [WORD_BITS-1:0] pivot_row_val;
    reg signed [2*WORD_BITS-1:0] prod;
    reg signed [2*WORD_BITS-1:0] delta_wide;
    reg signed [WORD_BITS-1:0]   delta;
    reg signed [WORD_BITS:0]     diff_wide;
    begin
        prod = factor * pivot_row_val;
        delta_wide = (prod + (1 <<< (FRAC_BITS-1))) >>> FRAC_BITS;
        if (delta_wide > Q88_MAX_C)
            delta = Q88_MAX_C;
        else if (delta_wide < Q88_MIN_C)
            delta = Q88_MIN_C;
        else
            delta = delta_wide[WORD_BITS-1:0];

        diff_wide = row_val - delta;
        if (diff_wide > Q88_MAX_C)
            eliminate_col = Q88_MAX_C;
        else if (diff_wide < Q88_MIN_C)
            eliminate_col = Q88_MIN_C;
        else
            eliminate_col = diff_wide[WORD_BITS-1:0];
    end
endfunction

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state    <= S_IDLE;
        busy     <= 1'b0;
        done     <= 1'b0;
        singular <= 1'b0;
        x_out    <= {(N*WORD_BITS){1'b0}};
        ainv_out <= {(N*N*WORD_BITS){1'b0}};
    end else begin
        done <= 1'b0;

        case (state)
        S_IDLE: begin
            if (start) begin
                busy     <= 1'b1;
                singular <= 1'b0;
                state    <= S_LOAD;
            end
        end

        S_LOAD: begin
            for (ii = 0; ii < N; ii = ii + 1) begin
                for (jj = 0; jj < N; jj = jj + 1) begin
                    aug[ii][jj]   <= a_in[(ii*N+jj+1)*WORD_BITS-1 -: WORD_BITS];
                    aug[ii][N+jj] <= (ii == jj) ? ONE_Q88 : {WORD_BITS{1'b0}};
                end
                aug[ii][2*N] <= b_in[(ii+1)*WORD_BITS-1 -: WORD_BITS];
            end
            k     <= 3'd0;
            state <= S_PIVOT_SEARCH;
        end

        S_PIVOT_SEARCH: begin
            best_row_v = k;
            best_val_v = aug[k][k];
            for (srch_i = 0; srch_i < N; srch_i = srch_i + 1) begin
                if (srch_i > k && abs_q88(aug[srch_i][k]) > abs_q88(best_val_v)) begin
                    best_val_v = aug[srch_i][k];
                    best_row_v = srch_i[2:0];
                end
            end
            pivot_row <= best_row_v;
            pivot_val <= best_val_v;
            if (abs_q88(best_val_v) < EPSILON_Q88) begin
                state <= S_SINGULAR_DONE;
            end else if (best_row_v != k) begin
                state <= S_PIVOT_SWAP;
            end else begin
                col_idx <= 3'd0;
                state   <= S_NORM_ISSUE;
            end
        end

        S_PIVOT_SWAP: begin
            for (jj = 0; jj < M; jj = jj + 1) begin
                aug[k][jj]         <= aug[pivot_row][jj];
                aug[pivot_row][jj] <= aug[k][jj];
            end
            col_idx <= 3'd0;
            state   <= S_NORM_ISSUE;
        end

        S_NORM_ISSUE: begin
            state <= S_NORM_WAIT;
        end

        S_NORM_WAIT: begin
            if (div_done) begin
                aug[k][col_idx] <= div_quotient;
                if (col_idx == M-1) begin
                    elim_idx <= 3'd0;
                    state    <= S_ELIM_ROW;
                end else begin
                    col_idx <= col_idx + 3'd1;
                    state   <= S_NORM_ISSUE;
                end
            end
        end

        S_ELIM_ROW: begin
            if (elim_idx == N-1 && elim_idx == k) begin
                state <= S_NEXT_COLUMN;
            end else if (elim_idx == k) begin
                elim_idx <= elim_idx + 3'd1;
            end else begin
                for (jj = 0; jj < M; jj = jj + 1) begin
                    aug[elim_idx][jj] <= eliminate_col(aug[elim_idx][jj], aug[elim_idx][k], aug[k][jj]);
                end
                if (elim_idx == N-1)
                    state <= S_NEXT_COLUMN;
                else
                    elim_idx <= elim_idx + 3'd1;
            end
        end

        S_NEXT_COLUMN: begin
            if (k == N-1) begin
                state <= S_DONE;
            end else begin
                k     <= k + 3'd1;
                state <= S_PIVOT_SEARCH;
            end
        end

        S_DONE: begin
            for (ii = 0; ii < N; ii = ii + 1) begin
                for (jj = 0; jj < N; jj = jj + 1) begin
                    ainv_out[(ii*N+jj+1)*WORD_BITS-1 -: WORD_BITS] <= aug[ii][N+jj];
                end
                x_out[(ii+1)*WORD_BITS-1 -: WORD_BITS] <= aug[ii][2*N];
            end
            done  <= 1'b1;
            busy  <= 1'b0;
            state <= S_IDLE;
        end

        S_SINGULAR_DONE: begin
            singular <= 1'b1;
            done     <= 1'b1;
            busy     <= 1'b0;
            x_out    <= {(N*WORD_BITS){1'b0}};
            ainv_out <= {(N*N*WORD_BITS){1'b0}};
            state    <= S_IDLE;
        end

        default: state <= S_IDLE;
        endcase
    end
end

endmodule
