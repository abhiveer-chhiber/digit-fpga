`timescale 1ns/1ps

module fixed_neuron_tb;
    logic clk = 0;
    always #5 clk = ~clk;

    logic reset = 1;
    logic start = 0;
    logic signed [31:0] bias = 0;
    logic relu_enable = 0;
    logic input_valid = 0;
    logic input_last = 0;
    logic signed [11:0] input_value = 0;
    logic signed [11:0] weight_value = 0;
    wire busy;
    wire output_valid;
    wire signed [11:0] output_value;
    wire accumulator_saturated;
    wire output_saturated;

    fixed_neuron dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .bias(bias),
        .relu_enable(relu_enable),
        .input_valid(input_valid),
        .input_last(input_last),
        .input_value(input_value),
        .weight_value(weight_value),
        .busy(busy),
        .output_valid(output_valid),
        .output_value(output_value),
        .accumulator_saturated(accumulator_saturated),
        .output_saturated(output_saturated)
    );

    integer file_handle;
    integer read_count;
    integer case_count;
    integer case_index;
    integer term_count;
    integer term_index;
    integer case_bias;
    integer case_relu;
    integer expected_value;
    integer expected_acc_clip;
    integer expected_output_clip;
    integer value;
    integer weight;

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    task automatic check_reset;
        if (busy !== 0 || output_valid !== 0 || output_value !== 0 ||
            accumulator_saturated !== 0 || output_saturated !== 0)
            $fatal(1, "Reset did not clear the neuron");
    endtask

    initial begin
        #50000000;
        $fatal(1, "Simulation timed out");
    end

    initial begin
        repeat (2) tick();
        check_reset();

        // Start a calculation, then reset before it finishes.
        @(negedge clk);
        reset = 0;
        start = 1;
        bias = 100;
        tick();
        if (busy !== 1)
            $fatal(1, "Start did not make the neuron busy");

        @(negedge clk);
        start = 0;
        input_valid = 1;
        input_value = 64;
        weight_value = 64;
        tick();

        @(negedge clk);
        reset = 1;
        tick();
        check_reset();

        @(negedge clk);
        reset = 0;
        input_valid = 0;

        file_handle = $fopen("build/sim/neuron-vectors.txt", "r");
        if (file_handle == 0)
            $fatal(1, "Cannot open neuron test vectors");

        read_count = $fscanf(file_handle, "%d", case_count);
        if (read_count != 1 || case_count < 1)
            $fatal(1, "Invalid case count");

        for (case_index = 0; case_index < case_count; case_index = case_index + 1) begin
            read_count = $fscanf(
                file_handle, "%d %d %d %d %d %d",
                term_count, case_bias, case_relu, expected_value,
                expected_acc_clip, expected_output_clip
            );
            if (read_count != 6 || term_count < 1)
                $fatal(1, "Invalid header for case %0d", case_index);

            @(negedge clk);
            start = 1;
            bias = case_bias;
            relu_enable = case_relu;
            input_valid = 0;
            input_last = 0;
            tick();

            if (busy !== 1 || output_valid !== 0 ||
                accumulator_saturated !== 0 || output_saturated !== 0)
                $fatal(1, "Incorrect start signals in case %0d", case_index);

            for (term_index = 0; term_index < term_count; term_index = term_index + 1) begin
                read_count = $fscanf(file_handle, "%d %d", value, weight);
                if (read_count != 2)
                    $fatal(1, "Missing input in case %0d", case_index);

                // Invalid cycles must not consume inputs, even with input_last high.
                if (term_index % 7 == 0) begin
                    @(negedge clk);
                    start = 0;
                    input_valid = 0;
                    input_last = 1;
                    input_value = -2048;
                    weight_value = -2048;
                    bias = 32'sh7fffffff;
                    relu_enable = !case_relu;
                    tick();
                    if (busy !== 1 || output_valid !== 0)
                        $fatal(1, "Invalid-cycle handling failed in case %0d", case_index);
                end

                @(negedge clk);
                // A new start while busy must not replace the current calculation.
                start = term_index == 2;
                input_valid = 1;
                input_last = term_index == term_count - 1;
                input_value = value;
                weight_value = weight;
                bias = 32'sh7fffffff;
                relu_enable = !case_relu;
                tick();

                if (busy !== 1 || output_valid !== 0)
                    $fatal(1, "Early completion in case %0d", case_index);
            end

            @(negedge clk);
            start = 0;
            input_valid = 0;
            input_last = 0;
            tick();

            if (output_valid !== 1 || busy !== 0)
                $fatal(1, "Missing completion in case %0d", case_index);
            if ($signed(output_value) !== expected_value)
                $fatal(1, "Case %0d: expected %0d, got %0d",
                    case_index, expected_value, $signed(output_value));
            if (accumulator_saturated !== expected_acc_clip[0])
                $fatal(1, "Accumulator flag mismatch in case %0d", case_index);
            if (output_saturated !== expected_output_clip[0])
                $fatal(1, "Output flag mismatch in case %0d", case_index);

            tick();
            if (output_valid !== 0)
                $fatal(1, "Completion lasted more than one cycle");
            if ($signed(output_value) !== expected_value)
                $fatal(1, "Output changed while idle");
        end

        $fclose(file_handle);
        $display("PASS: %0d neuron cases matched Python outputs and saturation flags", case_count);
        $display("PASS: reset, input gaps, saved bias/ReLU, busy-start handling, and completion pulse");
        $finish;
    end
endmodule
