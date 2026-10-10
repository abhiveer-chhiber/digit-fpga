`timescale 1ns/1ps

module fixed_neuron (
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic signed [31:0] bias,
    input  logic relu_enable,
    input  logic input_valid,
    input  logic input_last,
    input  logic signed [11:0] input_value,
    input  logic signed [11:0] weight_value,
    output logic busy,
    output logic output_valid,
    output logic signed [11:0] output_value,
    output logic accumulator_saturated,
    output logic output_saturated
);

    // Inputs and weights have 6 fractional bits; products and bias have 12.
    typedef enum logic [1:0] {IDLE, MAC, FINISH} state_t;
    state_t state;

    logic signed [31:0] accumulator;
    logic signed [31:0] saved_bias;
    logic saved_relu;

    logic signed [23:0] product;
    logic signed [32:0] mac_sum;
    logic signed [32:0] bias_sum;
    logic signed [31:0] final_accumulator;
    logic signed [32:0] magnitude;
    logic signed [32:0] rounded;

    localparam logic signed [32:0] ACC_MAX = 33'sd2147483647;
    localparam logic signed [32:0] ACC_MIN = -33'sd2147483648;
    localparam logic signed [32:0] OUTPUT_MAX = 33'sd2047;
    localparam logic signed [32:0] OUTPUT_MIN = -33'sd2048;

    function automatic logic signed [31:0] clip_accumulator(
        input logic signed [32:0] value
    );
        if (value > ACC_MAX)
            clip_accumulator = 32'sh7fffffff;
        else if (value < ACC_MIN)
            clip_accumulator = 32'sh80000000;
        else
            clip_accumulator = value[31:0];
    endfunction

    always @* begin
        busy = state != IDLE;
        product = input_value * weight_value;

        mac_sum = $signed({accumulator[31], accumulator})
                + $signed({{9{product[23]}}, product});
        bias_sum = $signed({accumulator[31], accumulator})
                 + $signed({saved_bias[31], saved_bias});

        final_accumulator = clip_accumulator(bias_sum);
        if (saved_relu && final_accumulator < 0)
            final_accumulator = 0;

        // Widen before taking the magnitude so the minimum signed value fits.
        magnitude = $signed({final_accumulator[31], final_accumulator});
        if (final_accumulator < 0)
            magnitude = -magnitude;

        // Round halfway values away from zero, then restore the sign.
        rounded = (magnitude + 33'sd32) >>> 6;
        if (final_accumulator < 0)
            rounded = -rounded;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            accumulator <= 0;
            saved_bias <= 0;
            saved_relu <= 0;
            output_valid <= 0;
            output_value <= 0;
            accumulator_saturated <= 0;
            output_saturated <= 0;
        end else begin
            output_valid <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        accumulator <= 0;
                        saved_bias <= bias;
                        saved_relu <= relu_enable;
                        accumulator_saturated <= 0;
                        output_saturated <= 0;
                        state <= MAC;
                    end
                end

                MAC: begin
                    if (input_valid) begin
                        accumulator <= clip_accumulator(mac_sum);
                        if (mac_sum > ACC_MAX || mac_sum < ACC_MIN)
                            accumulator_saturated <= 1;
                        if (input_last)
                            state <= FINISH;
                    end
                end

                FINISH: begin
                    if (bias_sum > ACC_MAX || bias_sum < ACC_MIN)
                        accumulator_saturated <= 1;

                    if (rounded > OUTPUT_MAX) begin
                        output_value <= 12'sh7ff;
                        output_saturated <= 1;
                    end else if (rounded < OUTPUT_MIN) begin
                        output_value <= 12'sh800;
                        output_saturated <= 1;
                    end else begin
                        output_value <= rounded[11:0];
                        output_saturated <= 0;
                    end

                    output_valid <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
