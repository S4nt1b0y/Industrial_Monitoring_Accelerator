module twiddle_lut #(
    parameter W    = 16,
    parameter AW   = 5,
    parameter HALF = 32
)(
    input  wire                  clk,
    input  wire [AW-1:0]         k,
    output wire signed [W-1:0]   w_re,
    output wire signed [W-1:0]   w_im
);

    // ============================================================
    // ROM:
    //
    //   32 palavras
    //   32 bits por palavra
    //
    //   [31:16] = W_RE
    //   [15: 0] = W_IM
    //
    // Implementação física desejada:
    //
    //   Cyclone V -> M10K
    //
    // Leitura síncrona:
    //
    //   k(t) -> M10K -> data(t+1)
    // ============================================================

    wire [31:0] rom_data;

    altsyncram #(
        .operation_mode("ROM"),

        .width_a(32),
        .widthad_a(AW),
        .numwords_a(HALF),

        .outdata_reg_a("CLOCK0"),

        .address_aclr_a("NONE"),
        .outdata_aclr_a("NONE"),

        .clock_enable_input_a("BYPASS"),
        .clock_enable_output_a("BYPASS"),

        .ram_block_type("M10K"),

        .init_file("twiddle.mif"),
        .init_file_layout("PORT_A"),

        .intended_device_family("Cyclone V"),

        .lpm_type("altsyncram")
    ) twiddle_rom (
        .clock0(clk),
        .address_a(k),
        .q_a(rom_data),

        .aclr0(1'b0),
        .aclr1(1'b0),

        .addressstall_a(1'b0),
        .clocken0(1'b1),
        .clocken1(1'b1),

        .eccstatus(),
        .rden_a(1'b1)
    );

    // ============================================================
    // Separação real / imaginário
    // ============================================================

    assign w_re = rom_data[31:16];
    assign w_im = rom_data[15:0];

endmodule
