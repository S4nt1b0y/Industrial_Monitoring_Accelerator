/*
 * Module: fft_datapath
 * Memories, counters and butterfly for the 64-point radix-2 DIF FFT.
 *
 * Two ping-pong banks (the problem statement allows at most two
 * internal memories): each stage reads one bank and writes the other.
 * Each bank is a ram of 64 words of {re, im}, with one registered read
 * and one write per cycle, so it maps to M10K. The banks have no reset:
 * loading writes all 64 words of bank 0, and every stage writes all 64
 * words of its destination before they are read.
 * With one read port per bank, operand a is fetched and held, then b is
 * fetched. The two results are written in two cycles.
 * Butterfly b of stage s works on the pair (base+j, base+j+half), with
 * half = 32>>s, j = b & (half-1) and base = (b & ~(half-1)) << 1.
 * The twiddle for stage s is entry j<<s of twiddle_rom.
 * The last stage leaves the result in bit-reversed order; the read port
 * reverses the 6 address bits so callers use natural bin order.
 * Read latency: one cycle.
 * Control pulses come from fft_ctrl.
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
    input  wire bf_fetch_a,
    input  wire bf_fetch_b,
    input  wire bf_issue,
    input  wire bf_write_a,
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

localparam CELL_BITS = 2*WORD_BITS;

reg [6:0] load_count;
reg [5:0] bf_index;
reg [2:0] stage;
reg       cur_bank;   // 0: source is bank 0, 1: source is bank 1

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

/* Read address, shared by both banks. Outside the two fetch cycles it
 * is the caller's bin, bit-reversed. */
wire [5:0] rev = {read_addr[0], read_addr[1], read_addr[2],
                  read_addr[3], read_addr[4], read_addr[5]};
wire [5:0] rd_addr = bf_fetch_a ? idx_a :
                     bf_fetch_b ? idx_b : rev;

/* Write port, shared by both banks: load goes to bank 0, butterfly
 * results go to the destination bank, a then b. */
wire [5:0] wr_addr = load_en    ? load_count[5:0] :
                     bf_write_a ? idx_a : idx_b;
wire signed [CELL_BITS-1:0] wr_data =
    load_en    ? {load_data, {WORD_BITS{1'b0}}} :
    bf_write_a ? {a_out_r, a_out_i} : {b_out_r, b_out_i};

wire load_write = load_en && !load_done;
wire bf_write   = bf_write_a || bf_store;

wire bank0_wr = load_write || (bf_write &&  cur_bank);
wire bank1_wr =               (bf_write && !cur_bank);

wire signed [CELL_BITS-1:0] bank0_rd, bank1_rd;

ram #(
    .WORD_BITS(CELL_BITS),
    .DEPTH(N),
    .ADDR_BITS(6)
) u_bank0 (
    .clk(clk),
    .wr_addr(wr_addr),
    .wr_data(wr_data),
    .wr_en(bank0_wr),
    .rd_addr(rd_addr),
    .rd_data(bank0_rd)
);

ram #(
    .WORD_BITS(CELL_BITS),
    .DEPTH(N),
    .ADDR_BITS(6)
) u_bank1 (
    .clk(clk),
    .wr_addr(wr_addr),
    .wr_data(wr_data),
    .wr_en(bank1_wr),
    .rd_addr(rd_addr),
    .rd_data(bank1_rd)
);

/* Operands come from the source bank. The result is in the bank the last
 * stage wrote; cur_bank still selects it because stage_next is not
 * pulsed after the last stage. */
wire signed [CELL_BITS-1:0] src_rd = cur_bank ? bank1_rd : bank0_rd;
wire signed [CELL_BITS-1:0] dst_rd = cur_bank ? bank0_rd : bank1_rd;

/* Operand a arrives during the fetch-b cycle and is held here. */
reg signed [CELL_BITS-1:0] a_hold;

fft_butterfly #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS)
) u_butterfly (
    .clk(clk),
    .rst_n(rst_n),
    .start(bf_issue),
    .ar(a_hold[CELL_BITS-1 -: WORD_BITS]),
    .ai(a_hold[WORD_BITS-1:0]),
    .br(src_rd[CELL_BITS-1 -: WORD_BITS]),
    .bi(src_rd[WORD_BITS-1:0]),
    .w_cos(w_cos),
    .w_sin(w_sin),
    .a_out_r(a_out_r),
    .a_out_i(a_out_i),
    .b_out_r(b_out_r),
    .b_out_i(b_out_i),
    .done(bf_done)
);

assign read_re = dst_rd[CELL_BITS-1 -: WORD_BITS];
assign read_im = dst_rd[WORD_BITS-1:0];

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        load_count <= 7'd0;
        bf_index   <= 6'd0;
        stage      <= 3'd0;
        cur_bank   <= 1'b0;
        a_hold     <= {CELL_BITS{1'b0}};
    end else begin
        if (run_init) begin
            load_count <= 7'd0;
            bf_index   <= 6'd0;
            stage      <= 3'd0;
            cur_bank   <= 1'b0;
        end

        if (load_write)
            load_count <= load_count + 7'd1;

        if (bf_fetch_b)
            a_hold <= src_rd;

        if (bf_store)
            bf_index <= bf_last ? 6'd0 : bf_index + 6'd1;

        if (stage_next) begin
            stage    <= stage + 3'd1;
            cur_bank <= ~cur_bank;
        end
    end
end

endmodule
