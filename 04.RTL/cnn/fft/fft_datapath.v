/*
 * Module: fft_datapath
 * Memories, counters and butterfly instance for the 64-point radix-2
 * DIF FFT. Two ping-pong banks (the problem statement allows at most
 * two internal memories): a stage reads one and writes the other, and
 * the next stage swaps, so no in-place read-after-write hazard exists.
 *
 * Butterfly b of stage s works on the pair (base+j, base+j+half) with
 * half = 32>>s, j = b & (half-1) and base = (b & ~(half-1)) << 1 --
 * both masks, since half is always a power of two.
 *
 * Twiddle: the stage needs exp(-j*2*pi*j/span) with span = 64>>s,
 * which is exactly entry j<<s of the 64-point table in twiddle_rom.
 *
 * The result leaves the last stage in bit-reversed order, as the
 * algorithm produces it; the read port reverses the 6 address bits
 * (pure wire reordering) so callers see natural bin order.
 *
 * Every state decision comes from fft_ctrl as a control pulse.
 * Status: implemented.
 */
module fft_datapath #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,
    parameter FRAC_BITS = 15,
    parameter N         = 64,
    parameter STAGES    = 6
) (
    input  wire clk,
    input  wire rst_n,

    input  wire run_init,
    input  wire load_en,
    input  wire bf_issue,
    input  wire bf_store,
    input  wire stage_next,

    input  wire signed [WORD_BITS-1:0] load_data,

    output wire load_done,
    output wire bf_done,
    output wire bf_last,
    output wire stage_last,

    input  wire [5:0]                  read_addr,
    output wire signed [WORD_BITS-1:0] read_re,
    output wire signed [WORD_BITS-1:0] read_im
);

integer i;

reg signed [WORD_BITS-1:0] mem_re [0:2*N-1];
reg signed [WORD_BITS-1:0] mem_im [0:2*N-1];

reg [6:0] load_count;
reg [5:0] bf_index;
reg [2:0] stage;
reg       cur_bank;   // 0: source is the low half, 1: source is the high half

wire [6:0] src_base = cur_bank ? N : 0;
wire [6:0] dst_base = cur_bank ? 0 : N;

assign load_done  = (load_count == N);
assign stage_last = (stage == STAGES-1);

/* half = 32 >> stage */
wire [5:0] half = 6'd32 >> stage;
wire [5:0] j    = bf_index & (half - 6'd1);
wire [5:0] grp  = (bf_index & ~(half - 6'd1)) << 1;
wire [5:0] idx_a = grp + j;
wire [5:0] idx_b = idx_a + half;

assign bf_last = (bf_index == N/2 - 1);

/* twiddle index j<<s covers 0..31 in steps of 2^s */
wire [4:0] tw_k = j << stage;
wire signed [WORD_BITS-1:0] w_cos, w_sin;

twiddle_rom #(
    .WORD_BITS(WORD_BITS)
) u_twiddle (
    .k(tw_k),
    .cos_out(w_cos),
    .sin_out(w_sin)
);

wire signed [WORD_BITS-1:0] a_out_r, a_out_i, b_out_r, b_out_i;

fft_butterfly #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS)
) u_butterfly (
    .clk(clk),
    .rst_n(rst_n),
    .start(bf_issue),
    .ar(mem_re[src_base + idx_a]),
    .ai(mem_im[src_base + idx_a]),
    .br(mem_re[src_base + idx_b]),
    .bi(mem_im[src_base + idx_b]),
    .w_cos(w_cos),
    .w_sin(w_sin),
    .a_out_r(a_out_r),
    .a_out_i(a_out_i),
    .b_out_r(b_out_r),
    .b_out_i(b_out_i),
    .done(bf_done)
);

/* Result sits in the bank the last stage WROTE, i.e. dst_base -- the
 * controller does not pulse stage_next after the final stage, so
 * cur_bank still selects that stage's source/destination pair.
 * Bit-reverse the read address so the caller indexes natural bin order. */
wire [6:0] result_base = dst_base;
wire [5:0] rev = {read_addr[0], read_addr[1], read_addr[2],
                  read_addr[3], read_addr[4], read_addr[5]};
assign read_re = mem_re[result_base + rev];
assign read_im = mem_im[result_base + rev];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 2*N; i = i + 1) begin
            mem_re[i] <= {WORD_BITS{1'b0}};
            mem_im[i] <= {WORD_BITS{1'b0}};
        end
        load_count <= 7'd0;
        bf_index   <= 6'd0;
        stage      <= 3'd0;
        cur_bank   <= 1'b0;
    end else begin
        if (run_init) begin
            load_count <= 7'd0;
            bf_index   <= 6'd0;
            stage      <= 3'd0;
            cur_bank   <= 1'b0;
        end

        /* load: real input, imaginary part zero, into the low bank */
        if (load_en && !load_done) begin
            mem_re[load_count[5:0]] <= load_data;
            mem_im[load_count[5:0]] <= {WORD_BITS{1'b0}};
            load_count <= load_count + 7'd1;
        end

        if (bf_store) begin
            mem_re[dst_base + idx_a] <= a_out_r;
            mem_im[dst_base + idx_a] <= a_out_i;
            mem_re[dst_base + idx_b] <= b_out_r;
            mem_im[dst_base + idx_b] <= b_out_i;
            bf_index <= bf_last ? 6'd0 : bf_index + 6'd1;
        end

        if (stage_next) begin
            stage    <= stage + 3'd1;
            cur_bank <= ~cur_bank;
        end
    end
end

endmodule
