module addr_gen #(
    parameter M = 6   // log2(N)
)(
    input  wire [M-1:0] stage,
    input  wire [M-2:0] j,
    output reg  [M-1:0] a_idx,
    output reg  [M-1:0] b_idx,
    output reg  [M-2:0] k
);
 
    reg [M-1:0] span, pos, group, base;
    reg [M-1:0] sh;                       // sh = (M-1) - stage
 
    always @(*) begin
        sh    = (M-1) - stage;            // estagio 0 -> M-1 ; ultimo -> 0
        span  = (1 << sh);                // N/2, N/4, ..., 1
        pos   = j & (span - 1);           // posicao dentro do grupo
        group = j >> sh;                  // indice do grupo
        base  = group << (sh + 1);        // group * 2*span
        a_idx = base + pos;
        b_idx = a_idx + span;
        k     = pos << stage;             // expoente do twiddle = pos * 2^stage
    end
 
endmodule