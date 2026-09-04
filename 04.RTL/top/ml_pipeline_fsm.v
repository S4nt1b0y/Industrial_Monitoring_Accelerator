/*
 * Module: ml_pipeline_fsm
 * Control FSM for the serialized ML feature pipeline.
 */
module ml_pipeline_fsm (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       valid_i,
    input  wire       fft_done_i,
    input  wire       mdc_pico_ready_i,
    input  wire       mdc_done_i,
    input  wire       classifier_valid_i,
    output wire       ready_o,
    output reg        capture_o,
    output reg        fft_start_o,
    output reg        mdc_start_o,
    output reg        pico_valid_o,
    output reg        store_fft_o,
    output reg        store_mdc_o,
    output reg        classifier_start_o,
    output reg [1:0]  channel_sel_o,
    output reg [1:0]  pico_sel_o,
    output reg [2:0]  stage_o
);

localparam [2:0] STAGE_IDLE        = 3'd0;
localparam [2:0] STAGE_PROCESS_X_A = 3'd1;
localparam [2:0] STAGE_PROCESS_X_B = 3'd2;
localparam [2:0] STAGE_PROCESS_Y_A = 3'd3;
localparam [2:0] STAGE_PROCESS_Y_B = 3'd4;
localparam [2:0] STAGE_CLASSIFY    = 3'd5;

localparam [1:0] CH_X_A = 2'd0;
localparam [1:0] CH_X_B = 2'd1;
localparam [1:0] CH_Y_A = 2'd2;
localparam [1:0] CH_Y_B = 2'd3;

localparam [3:0] S_IDLE          = 4'd0;
localparam [3:0] S_START_FFT     = 4'd1;
localparam [3:0] S_WAIT_FFT      = 4'd2;
localparam [3:0] S_SEND_PICO_0   = 4'd3;
localparam [3:0] S_SEND_PICO_1   = 4'd4;
localparam [3:0] S_SEND_PICO_2   = 4'd5;
localparam [3:0] S_WAIT_MDC      = 4'd6;
localparam [3:0] S_CLASSIFY      = 4'd7;
localparam [3:0] S_WAIT_CLASSIFY = 4'd8;

reg [3:0] state;
reg [3:0] state_next;
reg [1:0] channel_sel_next;

assign ready_o = (state == S_IDLE);

always @(*) begin
    capture_o          = 1'b0;
    fft_start_o        = 1'b0;
    mdc_start_o        = 1'b0;
    pico_valid_o       = 1'b0;
    store_fft_o        = 1'b0;
    store_mdc_o        = 1'b0;
    classifier_start_o = 1'b0;
    pico_sel_o         = 2'd0;
    state_next         = state;
    channel_sel_next   = channel_sel_o;

    case (state)
        S_IDLE: begin
            channel_sel_next = CH_X_A;
            if (valid_i) begin
                capture_o  = 1'b1;
                state_next = S_START_FFT;
            end
        end

        S_START_FFT: begin
            fft_start_o = 1'b1;
            state_next  = S_WAIT_FFT;
        end

        S_WAIT_FFT: begin
            if (fft_done_i) begin
                store_fft_o = 1'b1;
                mdc_start_o = 1'b1;
                state_next  = S_SEND_PICO_0;
            end
        end

        S_SEND_PICO_0: begin
            pico_sel_o   = 2'd0;
            pico_valid_o = mdc_pico_ready_i;
            if (mdc_pico_ready_i) begin
                state_next = S_SEND_PICO_1;
            end
        end

        S_SEND_PICO_1: begin
            pico_sel_o   = 2'd1;
            pico_valid_o = mdc_pico_ready_i;
            if (mdc_pico_ready_i) begin
                state_next = S_SEND_PICO_2;
            end
        end

        S_SEND_PICO_2: begin
            pico_sel_o   = 2'd2;
            pico_valid_o = mdc_pico_ready_i;
            if (mdc_pico_ready_i) begin
                state_next = S_WAIT_MDC;
            end
        end

        S_WAIT_MDC: begin
            if (mdc_done_i) begin
                store_mdc_o = 1'b1;
                if (channel_sel_o == CH_Y_B) begin
                    state_next = S_CLASSIFY;
                end else begin
                    channel_sel_next = channel_sel_o + 2'd1;
                    state_next       = S_START_FFT;
                end
            end
        end

        S_CLASSIFY: begin
            classifier_start_o = 1'b1;
            state_next         = S_WAIT_CLASSIFY;
        end

        S_WAIT_CLASSIFY: begin
            if (classifier_valid_i) begin
                state_next = S_IDLE;
            end
        end

        default: begin
            state_next       = S_IDLE;
            channel_sel_next = CH_X_A;
        end
    endcase
end

always @(*) begin
    case (state)
        S_IDLE:          stage_o = STAGE_IDLE;
        S_CLASSIFY,
        S_WAIT_CLASSIFY: stage_o = STAGE_CLASSIFY;
        default: begin
            case (channel_sel_o)
                CH_X_A:  stage_o = STAGE_PROCESS_X_A;
                CH_X_B:  stage_o = STAGE_PROCESS_X_B;
                CH_Y_A:  stage_o = STAGE_PROCESS_Y_A;
                CH_Y_B:  stage_o = STAGE_PROCESS_Y_B;
                default: stage_o = STAGE_IDLE;
            endcase
        end
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state         <= S_IDLE;
        channel_sel_o <= CH_X_A;
    end else begin
        state         <= state_next;
        channel_sel_o <= channel_sel_next;
    end
end

endmodule
