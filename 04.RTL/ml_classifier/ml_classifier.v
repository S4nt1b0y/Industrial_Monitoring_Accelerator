/*
 * Module: ml_classifier
 * Decision-tree classifier for motor state detection from Q1.15 FFT/MDC features.
 *
 * Input feature packing:
 *   features_i[15:0]       = feature 0
 *   features_i[(N*16)+:16] = feature N
 *
 * Feature map used by this tree:
 *   8   = aceleracao_x_mancal_a, FFT bin 8
 *   9   = aceleracao_x_mancal_a, FFT bin 9
 *   36  = aceleracao_x_mancal_b, FFT bin 3
 *   41  = aceleracao_x_mancal_b, FFT bin 8
 *   56  = aceleracao_x_mancal_b, FFT bin 23
 *   87  = aceleracao_y_mancal_a, FFT bin 21
 *   89  = aceleracao_y_mancal_a, FFT bin 23
 *   102 = aceleracao_y_mancal_b, FFT bin 3
 *   122 = aceleracao_y_mancal_b, FFT bin 23
 *
 * Class map:
 *   0 = operacao_normal
 *   1 = desalinhamento
 *   2 = desbalanceamento
 *   3 = desgaste_rolamento
 *
 * Tree source:
 *   03.Reference/artifacts/dataset_evaluation/motor_measurements_q15/lms_off_mdc_on/tree_q1_15.json
 */
module ml_classifier (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_i,
    input  wire [2239:0] features_i,
    output reg          valid_o,
    output reg  [1:0]   class_o
);

localparam [1:0] CLASS_NORMAL        = 2'd0;
localparam [1:0] CLASS_MISALIGNMENT  = 2'd1;
localparam [1:0] CLASS_UNBALANCE     = 2'd2;
localparam [1:0] CLASS_BEARING_WEAR  = 2'd3;

wire [15:0] feature_8;
wire [15:0] feature_9;
wire [15:0] feature_36;
wire [15:0] feature_41;
wire [15:0] feature_56;
wire [15:0] feature_87;
wire [15:0] feature_89;
wire [15:0] feature_102;
wire [15:0] feature_122;

reg [1:0] class_next;

assign feature_8   = features_i[(8   * 16) +: 16];
assign feature_9   = features_i[(9   * 16) +: 16];
assign feature_36  = features_i[(36  * 16) +: 16];
assign feature_41  = features_i[(41  * 16) +: 16];
assign feature_56  = features_i[(56  * 16) +: 16];
assign feature_87  = features_i[(87  * 16) +: 16];
assign feature_89  = features_i[(89  * 16) +: 16];
assign feature_102 = features_i[(102 * 16) +: 16];
assign feature_122 = features_i[(122 * 16) +: 16];

always @(*) begin
    class_next = CLASS_NORMAL;

    /*
     * Internal node rule:
     *   if feature <= threshold_q15, take the left child;
     *   otherwise, take the right child.
     */
    if (feature_102 <= 16'd666) begin
        /* Node 1: feature[9] <= 86 */
        if (feature_9 <= 16'd86) begin
            /* Node 2: feature[122] <= 1980 */
            if (feature_122 <= 16'd1980) begin
                /* Node 3: feature[122] <= 1266 */
                if (feature_122 <= 16'd1266) begin
                    /* Node 4: feature[122] <= 900 */
                    if (feature_122 <= 16'd900) begin
                        class_next = CLASS_NORMAL;
                    end else begin
                        class_next = CLASS_NORMAL;
                    end
                end else begin
                    /* Node 7: feature[8] <= 74 */
                    if (feature_8 <= 16'd74) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end else begin
                /* Node 10: feature[56] <= 30 */
                if (feature_56 <= 16'd30) begin
                    /* Node 11: feature[8] <= 66 */
                    if (feature_8 <= 16'd66) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 14: feature[8] <= 40 */
                    if (feature_8 <= 16'd40) begin
                        class_next = CLASS_MISALIGNMENT;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end
        end else begin
            /* Node 17: feature[122] <= 1396 */
            if (feature_122 <= 16'd1396) begin
                /* Node 18: feature[9] <= 134 */
                if (feature_9 <= 16'd134) begin
                    /* Node 19: feature[41] <= 54 */
                    if (feature_41 <= 16'd54) begin
                        class_next = CLASS_NORMAL;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 22: feature[9] <= 146 */
                    if (feature_9 <= 16'd146) begin
                        class_next = CLASS_MISALIGNMENT;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end else begin
                /* Node 25: feature[9] <= 104 */
                if (feature_9 <= 16'd104) begin
                    /* Node 26: feature[89] <= 390 */
                    if (feature_89 <= 16'd390) begin
                        class_next = CLASS_MISALIGNMENT;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 29: feature[87] <= 251 */
                    if (feature_87 <= 16'd251) begin
                        class_next = CLASS_MISALIGNMENT;
                    end else begin
                        class_next = CLASS_BEARING_WEAR;
                    end
                end
            end
        end
    end else begin
        /* Node 32: feature[36] <= 92 */
        if (feature_36 <= 16'd92) begin
            class_next = CLASS_BEARING_WEAR;
        end else begin
            class_next = CLASS_BEARING_WEAR;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_o <= 1'b0;
        class_o <= CLASS_NORMAL;
    end else begin
        valid_o <= valid_i;
        if (valid_i) begin
            class_o <= class_next;
        end
    end
end

endmodule
