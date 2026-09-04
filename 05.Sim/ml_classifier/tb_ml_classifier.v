`timescale 1ns/1ps

module tb_ml_classifier;

reg          clk;
reg          rst_n;
reg          valid_i;
reg [2239:0] features_i;
wire         valid_o;
wire [1:0]   class_o;

integer failures;

ml_classifier dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_i(valid_i),
    .features_i(features_i),
    .valid_o(valid_o),
    .class_o(class_o)
);

always #5 clk = ~clk;

task set_feature;
    input integer index;
    input [15:0] value;
    begin
        features_i[(index * 16) +: 16] = value;
    end
endtask

task run_case;
    input [127:0] name;
    input [1:0] expected_class;
    begin
        @(negedge clk);
        valid_i = 1'b1;
        @(posedge clk);
        #1;

        if (valid_o !== 1'b1 || class_o !== expected_class) begin
            $display(
                "FAIL %0s: valid_o=%b class_o=%0d expected=%0d",
                name,
                valid_o,
                class_o,
                expected_class
            );
            failures = failures + 1;
        end else begin
            $display("PASS %0s: class_o=%0d", name, class_o);
        end

        @(negedge clk);
        valid_i = 1'b0;
        features_i = {2240{1'b0}};
        @(posedge clk);
        #1;
    end
endtask

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    valid_i = 1'b0;
    features_i = {2240{1'b0}};
    failures = 0;

    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    /* Node path: 0L, 1L, 2L, 3L, 4L -> class 0 */
    set_feature(102, 16'd0);
    set_feature(122, 16'd0);
    set_feature(9,   16'd0);
    run_case("normal", 2'd0);

    /* Node path: 0L, 1R, 17L, 18L, 19R -> class 1 */
    set_feature(102, 16'd0);
    set_feature(122, 16'd0);
    set_feature(9,   16'd87);
    set_feature(41,  16'd55);
    run_case("misalignment", 2'd1);

    /* Node path: 0L, 1L, 2L, 3R, 7L -> class 2 */
    set_feature(102, 16'd0);
    set_feature(122, 16'd1267);
    set_feature(9,   16'd0);
    set_feature(8,   16'd74);
    run_case("unbalance", 2'd2);

    /* Node path: 0R, 32L -> class 3 */
    set_feature(102, 16'd667);
    set_feature(36,  16'd0);
    run_case("bearing_wear", 2'd3);

    if (failures == 0) begin
        $display("All ml_classifier tests passed.");
    end else begin
        $display("%0d ml_classifier test(s) failed.", failures);
    end

    $finish;
end

endmodule
