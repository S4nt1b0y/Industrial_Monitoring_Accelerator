/*
 * Module: ml_classifier
 * Decision-tree classifier for motor state detection from Q1.15 FFT features.
 *
 * Input feature packing:
 *   features_i[15:0]           = feature 0
 *   features_i[(N*16)+:16]     = feature N
 *
 * Feature map used by this tree:
 *   8   = aceleracao_x_mancal_a, FFT bin 8
 *   9   = aceleracao_x_mancal_a, FFT bin 9
 *   15  = aceleracao_x_mancal_a, FFT bin 15
 *   56  = aceleracao_y_mancal_a, FFT bin 23
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
 *   03.Reference/artifacts/ml_classifier/motor_measurements_q15/tree_q15.json
 */
module ml_classifier (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_i,
    input  wire [2111:0] features_i,
    output reg          valid_o,
    output reg  [1:0]   class_o
);

localparam [1:0] CLASS_NORMAL        = 2'd0;
localparam [1:0] CLASS_MISALIGNMENT  = 2'd1;
localparam [1:0] CLASS_UNBALANCE     = 2'd2;
localparam [1:0] CLASS_BEARING_WEAR  = 2'd3;

wire [15:0] feature_8;
wire [15:0] feature_9;
wire [15:0] feature_15;
wire [15:0] feature_56;
wire [15:0] feature_102;
wire [15:0] feature_122;

reg [1:0] class_next;

assign feature_8   = features_i[(8   * 16) +: 16];
assign feature_9   = features_i[(9   * 16) +: 16];
assign feature_15  = features_i[(15  * 16) +: 16];
assign feature_56  = features_i[(56  * 16) +: 16];
assign feature_102 = features_i[(102 * 16) +: 16];
assign feature_122 = features_i[(122 * 16) +: 16];

always @(*) begin
    class_next = CLASS_NORMAL;

    /*
     * Internal node rule:
     *   if feature <= threshold_q15, take the left child;
     *   otherwise, take the right child.
     */
    if (feature_102 <= 16'd568) begin
        /* Node 1: feature[122] <= 1854 */
        if (feature_122 <= 16'd1854) begin
            /* Node 2: feature[9] <= 100 */
            if (feature_9 <= 16'd100) begin
                /* Node 3: feature[122] <= 1286 */
                if (feature_122 <= 16'd1286) begin
                    /* Node 4: feature[15] <= 36 */
                    if (feature_15 <= 16'd36) begin
                        class_next = CLASS_NORMAL;
                    end else begin
                        class_next = CLASS_NORMAL;
                    end
                end else begin
                    /* Node 7: feature[8] <= 68 */
                    if (feature_8 <= 16'd68) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end else begin
                /* Node 10: feature[122] <= 1315 */
                if (feature_122 <= 16'd1315) begin
                    /* Node 11: feature[9] <= 140 */
                    if (feature_9 <= 16'd140) begin
                        class_next = CLASS_NORMAL;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 14: feature[56] <= 16 */
                    if (feature_56 <= 16'd16) begin
                        class_next = CLASS_NORMAL;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end
        end else begin
            /* Node 17: feature[9] <= 56 */
            if (feature_9 <= 16'd56) begin
                /* Node 18: feature[8] <= 44 */
                if (feature_8 <= 16'd44) begin
                    /* Node 19: feature[56] <= 32 */
                    if (feature_56 <= 16'd32) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 22: feature[56] <= 22 */
                    if (feature_56 <= 16'd22) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end else begin
                /* Node 25: feature[56] <= 20 */
                if (feature_56 <= 16'd20) begin
                    /* Node 26: feature[9] <= 100 */
                    if (feature_9 <= 16'd100) begin
                        class_next = CLASS_UNBALANCE;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end else begin
                    /* Node 29: feature[9] <= 76 */
                    if (feature_9 <= 16'd76) begin
                        class_next = CLASS_MISALIGNMENT;
                    end else begin
                        class_next = CLASS_MISALIGNMENT;
                    end
                end
            end
        end
    end else begin
        /* Node 32: feature[102] <= 671 */
        if (feature_102 <= 16'd671) begin
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
