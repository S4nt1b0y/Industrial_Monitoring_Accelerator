/*
 * Module: ml_pipeline
 * Serialized FFT/MDC feature pipeline for four acceleration channels.
 */
module ml_pipeline #(
    parameter DATA_WIDTH    = 16,
    parameter N             = 64,
    parameter M             = 6,
    parameter FRACW         = 15,
    parameter FS_HZ         = 6400,
    parameter MIN_K         = 2,
    parameter FFT_BIN_COUNT = 33,
    parameter NUM_FEATURES  = 140
)(
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire signed [N*DATA_WIDTH-1:0] acc_x_a_i,
    input  wire signed [N*DATA_WIDTH-1:0] acc_x_b_i,
    input  wire signed [N*DATA_WIDTH-1:0] acc_y_a_i,
    input  wire signed [N*DATA_WIDTH-1:0] acc_y_b_i,
    input  wire                            valid_i,
    output wire                            ready_o,
    output wire                            valid_o,
    output wire [1:0]                      class_o
);

localparam FEATURE_WIDTH = DATA_WIDTH * NUM_FEATURES;
localparam CHANNEL_COUNT = 4;
localparam FFT_FEATURES  = CHANNEL_COUNT * FFT_BIN_COUNT;
localparam [1:0] CH_X_A  = 2'd0;
localparam [1:0] CH_X_B  = 2'd1;
localparam [1:0] CH_Y_A  = 2'd2;
localparam [1:0] CH_Y_B  = 2'd3;

reg signed [N*DATA_WIDTH-1:0] acc_x_a_reg;
reg signed [N*DATA_WIDTH-1:0] acc_x_b_reg;
reg signed [N*DATA_WIDTH-1:0] acc_y_a_reg;
reg signed [N*DATA_WIDTH-1:0] acc_y_b_reg;
reg [FEATURE_WIDTH-1:0]       features;

wire       fsm_capture;
wire       fft_start;
wire       fft_done;
wire       mdc_start;
wire       mdc_done;
wire       mdc_result_valid;
wire       store_fft;
wire       store_mdc;
wire       classifier_start;
wire       classifier_valid;
wire [1:0] channel_sel;
wire [2:0] stage;

reg signed [N*DATA_WIDTH-1:0]  fft_in_re;
wire signed [N*DATA_WIDTH-1:0] fft_in_im;
wire signed [N*DATA_WIDTH-1:0] fft_out_re;
wire signed [N*DATA_WIDTH-1:0] fft_out_im;

reg [M-1:0]  peak_0;
reg [M-1:0]  peak_1;
reg [M-1:0]  peak_2;
wire [M-1:0] mdc_k0;
wire [31:0]  mdc_f0;

integer i;
integer insert_pos;
reg [DATA_WIDTH-1:0] mag_i;
reg [DATA_WIDTH-1:0] top_mag_0;
reg [DATA_WIDTH-1:0] top_mag_1;
reg [DATA_WIDTH-1:0] top_mag_2;
reg [M-1:0] top_idx_0;
reg [M-1:0] top_idx_1;
reg [M-1:0] top_idx_2;

assign fft_in_im = {N*DATA_WIDTH{1'b0}};

always @(*) begin
    case (channel_sel)
        CH_X_A:  fft_in_re = acc_x_a_reg;
        CH_X_B:  fft_in_re = acc_x_b_reg;
        CH_Y_A:  fft_in_re = acc_y_a_reg;
        CH_Y_B:  fft_in_re = acc_y_b_reg;
        default: fft_in_re = {N*DATA_WIDTH{1'b0}};
    endcase
end

function [DATA_WIDTH-1:0] abs_signed;
    input signed [DATA_WIDTH-1:0] value;
    begin
        if (value == {1'b1, {(DATA_WIDTH-1){1'b0}}}) begin
            abs_signed = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        end else if (value[DATA_WIDTH-1]) begin
            abs_signed = -value;
        end else begin
            abs_signed = value;
        end
    end
endfunction

function [DATA_WIDTH-1:0] fft_magnitude; 
    input signed [DATA_WIDTH-1:0] real_value;
    input signed [DATA_WIDTH-1:0] imag_value;
    reg [DATA_WIDTH:0] sum;
    begin
        sum = {1'b0, abs_signed(real_value)} + {1'b0, abs_signed(imag_value)};
        if (sum[DATA_WIDTH] || sum[DATA_WIDTH-1]) begin
            fft_magnitude = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        end else begin
            fft_magnitude = sum[DATA_WIDTH-1:0];
        end
    end
endfunction

function [DATA_WIDTH-1:0] saturate_f0;
    input [31:0] value;
    begin
        if (value > {{(32-DATA_WIDTH){1'b0}}, {1'b0, {(DATA_WIDTH-1){1'b1}}}}) begin
            saturate_f0 = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        end else begin
            saturate_f0 = value[DATA_WIDTH-1:0];
        end
    end
endfunction

// PENSO QUE DAQUI ATÉ LINA 236 mais ou menos será o modulo dedicado de obtenção de 
task update_top3;
    input [DATA_WIDTH-1:0] mag;
    input [M-1:0] idx;
    begin
        if ((mag > top_mag_0) || ((mag == top_mag_0) && (idx < top_idx_0))) begin
            top_mag_2 = top_mag_1;
            top_idx_2 = top_idx_1;
            top_mag_1 = top_mag_0;
            top_idx_1 = top_idx_0;
            top_mag_0 = mag;
            top_idx_0 = idx;
        end else if ((mag > top_mag_1) || ((mag == top_mag_1) && (idx < top_idx_1))) begin
            top_mag_2 = top_mag_1;
            top_idx_2 = top_idx_1;
            top_mag_1 = mag;
            top_idx_1 = idx;
        end else if ((mag > top_mag_2) || ((mag == top_mag_2) && (idx < top_idx_2))) begin
            top_mag_2 = mag;
            top_idx_2 = idx;
        end
    end
endtask

task sort_three_indices;
    input [M-1:0] idx_a;
    input [M-1:0] idx_b;
    input [M-1:0] idx_c;
    output [M-1:0] out_a;
    output [M-1:0] out_b;
    output [M-1:0] out_c;
    reg [M-1:0] tmp;
    begin
        out_a = idx_a;
        out_b = idx_b;
        out_c = idx_c;
        if (out_a > out_b) begin
            tmp = out_a;
            out_a = out_b;
            out_b = tmp;
        end
        if (out_b > out_c) begin
            tmp = out_b;
            out_b = out_c;
            out_c = tmp;
        end
        if (out_a > out_b) begin
            tmp = out_a;
            out_a = out_b;
            out_b = tmp;
        end
    end
endtask

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        acc_x_a_reg <= {N*DATA_WIDTH{1'b0}};
        acc_x_b_reg <= {N*DATA_WIDTH{1'b0}};
        acc_y_a_reg <= {N*DATA_WIDTH{1'b0}};
        acc_y_b_reg <= {N*DATA_WIDTH{1'b0}};
        features    <= {FEATURE_WIDTH{1'b0}};
        peak_0      <= {M{1'b0}};
        peak_1      <= {M{1'b0}};
        peak_2      <= {M{1'b0}};
    end else begin
        if (fsm_capture) begin
            acc_x_a_reg <= acc_x_a_i;
            acc_x_b_reg <= acc_x_b_i;
            acc_y_a_reg <= acc_y_a_i;
            acc_y_b_reg <= acc_y_b_i;
            features    <= {FEATURE_WIDTH{1'b0}};
        end

        if (store_fft) begin
            top_mag_0 = {DATA_WIDTH{1'b0}};
            top_mag_1 = {DATA_WIDTH{1'b0}};
            top_mag_2 = {DATA_WIDTH{1'b0}};
            top_idx_0 = {M{1'b1}};
            top_idx_1 = {M{1'b1}};
            top_idx_2 = {M{1'b1}};

            for (i = 0; i < FFT_BIN_COUNT; i = i + 1) begin
                mag_i = fft_magnitude(
                    fft_out_re[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH],
                    fft_out_im[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH]
                );
                features[((channel_sel * FFT_BIN_COUNT + i) * DATA_WIDTH) +: DATA_WIDTH] <= mag_i;
                update_top3(mag_i, i[M-1:0]);
            end

            sort_three_indices(top_idx_0, top_idx_1, top_idx_2, peak_0, peak_1, peak_2);
        end

        if (store_mdc) begin
            insert_pos = FFT_FEATURES + (channel_sel * 2);
            features[(insert_pos * DATA_WIDTH) +: DATA_WIDTH] <= saturate_f0(mdc_f0);
            features[((insert_pos + 1) * DATA_WIDTH) +: DATA_WIDTH] <= {{(DATA_WIDTH-1){1'b0}}, mdc_result_valid};
        end
    end
end

ml_pipeline_fsm u_fsm (
    .clk(clk),
    .rst_n(rst_n),
    .valid_i(valid_i),
    .fft_done_i(fft_done),
    .mdc_done_i(mdc_done),
    .classifier_valid_i(classifier_valid),
    .ready_o(ready_o),
    .capture_o(fsm_capture),
    .fft_start_o(fft_start),
    .mdc_start_o(mdc_start),
    .store_fft_o(store_fft),
    .store_mdc_o(store_mdc),
    .classifier_start_o(classifier_start),
    .channel_sel_o(channel_sel),
    .stage_o(stage)
);

fftu_dif #(
    .N(N),
    .M(M),
    .W(DATA_WIDTH),
    .FRACW(FRACW),
    .BF_LAT(2)
) u_fft (
    .clk(clk),
    .rst_n(rst_n),
    .start(fft_start),
    .in_re(fft_in_re),
    .in_im(fft_in_im),
    .done(fft_done),
    .out_re(fft_out_re),
    .out_im(fft_out_im)
);

mdc_tres_picos #(
    .WIDTH(M),
    .N_FFT(N)
) u_mdc (
    .clk(clk),
    .reset(~rst_n),
    .start(mdc_start),
    .pico1_i(peak_0),
    .pico2_i(peak_1),
    .pico3_i(peak_2),
    .fs_hz(FS_HZ),
    .min_k(MIN_K[M-1:0]),
    .k0(mdc_k0),
    .f0(mdc_f0),
    .busy(),
    .done(mdc_done),
    .result_valid(mdc_result_valid)
);

ml_classifier u_classifier (
    .clk(clk),
    .rst_n(rst_n),
    .valid_i(classifier_start),
    .features_i(features),
    .valid_o(classifier_valid),
    .class_o(class_o)
);

assign valid_o = classifier_valid;

endmodule
