
module fftu_dif #(
    parameter N     = 64,   // tamanho da FFT
    parameter M     = 6,    // log2(N)  -> N = 2^M
    parameter W     = 16,   // largura da palavra (Q1.FRACW)
    parameter FRACW = 15,   // bits fracionarios
    parameter BF_LAT = 2    // latencia do pipeline da borboleta
)(
    input  wire                    clk, rst_n,
    input  wire                    start,
    input  wire signed [N*W-1:0]   in_re, in_im,   // flattened: elem i = bits [(i+1)*W-1 -: W]
    output reg                     done,
    output wire signed [N*W-1:0]   out_re, out_im  // flattened da mesma forma
);

    // --------------------------------------------------------
    // Bancos de memoria (ping-pong)
    // --------------------------------------------------------
    reg signed [W-1:0] bank_re [0:1][0:N-1];
    reg signed [W-1:0] bank_im [0:1][0:N-1];

    reg  src;
    wire dst = ~src;

    reg [M-1:0]   stage;
    reg [M-2:0]   jcnt;

    // --------------------------------------------------------
    // Estados da FSM
    // --------------------------------------------------------
    parameter [2:0] IDLE  = 3'd0,
                     LOAD  = 3'd1,
                     RUN   = 3'd2,
                     FLUSH = 3'd3,
                     DONE  = 3'd4;
    reg [2:0] state;

    // --------------------------------------------------------
    // Bit-reversal
    // --------------------------------------------------------
    function [M-1:0] brev;
        input [M-1:0] x;
        integer bi;
        begin
            for (bi = 0; bi < M; bi = bi + 1)
                brev[bi] = x[M-1-bi];
        end
    endfunction

    // --------------------------------------------------------
    // Geracao de enderecos / twiddle / borboleta
    // --------------------------------------------------------
    wire [M-1:0] a_idx, b_idx;
    wire [M-2:0] k;
    wire signed [W-1:0] w_re, w_im;
    wire signed [W-1:0] x0_re, x0_im, x1_re, x1_im;

    butterfly_dif #(
        .W(W), .FRACW(FRACW)
    ) u_bf (
        .clk(clk), .rst_n(rst_n),
        .a_re(bank_re[src][a_idx]), .a_im(bank_im[src][a_idx]),
        .b_re(bank_re[src][b_idx]), .b_im(bank_im[src][b_idx]),
        .w_re(w_re), .w_im(w_im),
        .x0_re(x0_re), .x0_im(x0_im),
        .x1_re(x1_re), .x1_im(x1_im)
    );

    addr_gen #(
        .M(M)
    )    u_addr (
        .stage(stage), .j(jcnt),
        .a_idx(a_idx), .b_idx(b_idx), .k(k)
    );

    twiddle_lut #(
        .W(W), .AW(M-1), .HALF(N/2)
    ) u_tw (
        .k(k), .w_re(w_re), .w_im(w_im)
    );

    // --------------------------------------------------------
    // Registradores de atraso (pipeline da borboleta)
    // --------------------------------------------------------
    reg [M-1:0] a_idx_d [0:BF_LAT-1];
    reg [M-1:0] b_idx_d [0:BF_LAT-1];
    reg         vld_d   [0:BF_LAT-1];

    integer d;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (d = 0; d < BF_LAT; d = d + 1) begin
                a_idx_d[d] <= 0; b_idx_d[d] <= 0; vld_d[d] <= 1'b0;
            end
        end else begin
            a_idx_d[0] <= a_idx;
            b_idx_d[0] <= b_idx;
            vld_d[0]   <= (state == RUN);
            for (d = 1; d < BF_LAT; d = d + 1) begin
                a_idx_d[d] <= a_idx_d[d-1];
                b_idx_d[d] <= b_idx_d[d-1];
                vld_d[d]   <= vld_d[d-1];
            end
        end
    end

    // --------------------------------------------------------
    // FSM principal
    // --------------------------------------------------------
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done  <= 1'b0; src <= 1'b0; stage <= 0; jcnt <= 0;
        end else begin
            if (vld_d[BF_LAT-1]) begin
                bank_re[dst][a_idx_d[BF_LAT-1]] <= x0_re;
                bank_im[dst][a_idx_d[BF_LAT-1]] <= x0_im;
                bank_re[dst][b_idx_d[BF_LAT-1]] <= x1_re;
                bank_im[dst][b_idx_d[BF_LAT-1]] <= x1_im;
            end

            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) state <= LOAD;
                end

                LOAD: begin
                    for (i = 0; i < N; i = i + 1) begin
                        bank_re[0][i] <= in_re[(i+1)*W-1 -: W];
                        bank_im[0][i] <= in_im[(i+1)*W-1 -: W];
                    end
                    src <= 1'b0; stage <= 0; jcnt <= 0; state <= RUN;
                end

                RUN: begin
                    if (jcnt == (N/2-1)) state <= FLUSH;
                    else jcnt <= jcnt + 1;
                end

                FLUSH: begin
                    if (!vld_d[0] && !vld_d[BF_LAT-1]) begin
                        jcnt <= 0;
                        src  <= ~src;   // resultados deste estagio ficam no novo src
                        if (stage == (M-1)) state <= DONE;
                        else begin stage <= stage + 1; state <= RUN; end
                    end
                end

                DONE: begin
                    done <= 1'b1;
                    if (start) begin done <= 1'b0; state <= LOAD; end
                end

                default: state <= IDLE;
            endcase
        end
    end

    // --------------------------------------------------------
    // Saida (bit-reversal), flattened
    // --------------------------------------------------------
    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_out
            assign out_re[(gi+1)*W-1 -: W] = (done) ? bank_re[src][brev(gi)] : {W{1'b0}};
            assign out_im[(gi+1)*W-1 -: W] = (done) ? bank_im[src][brev(gi)] : {W{1'b0}};
        end
    endgenerate

endmodule