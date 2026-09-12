/*
 * Module: fir_decim
 * One polyphase FIR decimation stage: wires fir_decim_ctrl (FSM) to
 * fir_decim_datapath (delay line, coefficient ROM, MAC lanes). No
 * logic of its own beyond that connection.
 * Status: implemented.
 */
module fir_decim #(
    parameter WORD_BITS = 16,  // Q1.15
    parameter ACC_BITS  = 32,  // Q2.30
    parameter FRAC_BITS = 15,
    parameter TAPS      = 32,
    parameter NUM_MACS  = 4,
    parameter DECIM     = 8
) (
    input  wire clk,
    input  wire rst_n,

    input  wire                        sample_valid,
    input  wire signed [WORD_BITS-1:0] sample_in,

    output wire                        busy,
    output wire signed [WORD_BITS-1:0] sample_out,
    output wire                        sample_out_valid
);

wire trigger, batch_last;
wire batch_init, batch_step, do_reduce, do_output;

fir_decim_ctrl u_ctrl (
    .clk(clk),
    .rst_n(rst_n),
    .trigger(trigger),
    .batch_last(batch_last),
    .busy(busy),
    .batch_init(batch_init),
    .batch_step(batch_step),
    .do_reduce(do_reduce),
    .do_output(do_output)
);

fir_decim_datapath #(
    .WORD_BITS(WORD_BITS),
    .ACC_BITS(ACC_BITS),
    .FRAC_BITS(FRAC_BITS),
    .TAPS(TAPS),
    .NUM_MACS(NUM_MACS),
    .DECIM(DECIM)
) u_datapath (
    .clk(clk),
    .rst_n(rst_n),
    .sample_valid(sample_valid),
    .sample_in(sample_in),
    .batch_init(batch_init),
    .batch_step(batch_step),
    .do_reduce(do_reduce),
    .do_output(do_output),
    .trigger(trigger),
    .batch_last(batch_last),
    .sample_out(sample_out),
    .sample_out_valid(sample_out_valid)
);

endmodule
