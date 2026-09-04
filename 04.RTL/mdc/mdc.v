/*
 * Module: mdc_tres_picos
 * Computes the greatest common divisor of three FFT peak bins.
 */
module mdc_tres_picos #(
    parameter WIDTH = 6,
    parameter N_FFT = 64
)(
    input  wire             clk,
    input  wire             reset,
    input  wire             start,

    input  wire [WIDTH-1:0] pico1_i,
    input  wire [WIDTH-1:0] pico2_i,
    input  wire [WIDTH-1:0] pico3_i,

    input  wire [31:0]      fs_hz,
    input  wire [WIDTH-1:0] min_k,

    output reg  [WIDTH-1:0] k0,
    output reg  [31:0]      f0,
    output reg              busy,
    output reg              done,
    output reg              result_valid
);

reg [WIDTH-1:0] pico1_reg;
reg [WIDTH-1:0] pico2_reg;
reg [WIDTH-1:0] pico3_reg;
reg [WIDTH-1:0] a;
reg [WIDTH-1:0] b;
reg [WIDTH-1:0] m;
reg [WIDTH-1:0] k;

localparam [3:0] IDLE       = 4'd0;
localparam [3:0] CHECK      = 4'd1;
localparam [3:0] LOAD_12    = 4'd2;
localparam [3:0] MDC_12     = 4'd3;
localparam [3:0] LOAD_3     = 4'd4;
localparam [3:0] MDC_3      = 4'd5;
localparam [3:0] VALIDATE   = 4'd6;
localparam [3:0] DONE_OK    = 4'd7;
localparam [3:0] DONE_ERROR = 4'd8;

reg [3:0] state;

wire [WIDTH+31:0] freq_mult;

assign freq_mult = k * fs_hz;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        pico1_reg    <= {WIDTH{1'b0}};
        pico2_reg    <= {WIDTH{1'b0}};
        pico3_reg    <= {WIDTH{1'b0}};
        a            <= {WIDTH{1'b0}};
        b            <= {WIDTH{1'b0}};
        m            <= {WIDTH{1'b0}};
        k            <= {WIDTH{1'b0}};
        k0           <= {WIDTH{1'b0}};
        f0           <= 32'd0;
        busy         <= 1'b0;
        done         <= 1'b0;
        result_valid <= 1'b0;
        state        <= IDLE;
    end else begin
        done         <= 1'b0;
        result_valid <= 1'b0;

        case (state)
            IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    pico1_reg <= pico1_i;
                    pico2_reg <= pico2_i;
                    pico3_reg <= pico3_i;
                    busy      <= 1'b1;
                    state     <= CHECK;
                end
            end

            CHECK: begin
                busy <= 1'b1;
                if ((pico1_reg == {WIDTH{1'b0}}) ||
                    (pico2_reg == {WIDTH{1'b0}}) ||
                    (pico3_reg == {WIDTH{1'b0}})) begin
                    k0    <= {WIDTH{1'b0}};
                    f0    <= 32'd0;
                    state <= DONE_ERROR;
                end else begin
                    state <= LOAD_12;
                end
            end

            LOAD_12: begin
                a     <= pico1_reg;
                b     <= pico2_reg;
                state <= MDC_12;
            end

            MDC_12: begin
                if (a == b) begin
                    m     <= a;
                    state <= LOAD_3;
                end else if (a > b) begin
                    a <= a - b;
                end else begin
                    b <= b - a;
                end
            end

            LOAD_3: begin
                a     <= m;
                b     <= pico3_reg;
                state <= MDC_3;
            end

            MDC_3: begin
                if (a == b) begin
                    k     <= a;
                    state <= VALIDATE;
                end else if (a > b) begin
                    a <= a - b;
                end else begin
                    b <= b - a;
                end
            end

            VALIDATE: begin
                if (k < min_k) begin
                    k0    <= {WIDTH{1'b0}};
                    f0    <= 32'd0;
                    state <= DONE_ERROR;
                end else begin
                    k0    <= k;
                    f0    <= freq_mult / N_FFT;
                    state <= DONE_OK;
                end
            end

            DONE_OK: begin
                busy         <= 1'b0;
                done         <= 1'b1;
                result_valid <= 1'b1;
                state        <= IDLE;
            end

            DONE_ERROR: begin
                busy         <= 1'b0;
                done         <= 1'b1;
                result_valid <= 1'b0;
                state        <= IDLE;
            end

            default: begin
                busy         <= 1'b0;
                done         <= 1'b0;
                result_valid <= 1'b0;
                state        <= IDLE;
            end
        endcase
    end
end

endmodule
