`timescale 1ns/1ps

module neuron_trace_tb;
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

    fixed_neuron dut (.*);

    integer trace_file;
    integer cycle = 0;

    always @(posedge clk) begin
        #1;
        $fdisplay(trace_file,
            "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            cycle, reset, start, input_valid, input_last,
            busy, output_valid, $signed(input_value),
            $signed(weight_value), $signed(dut.accumulator),
            $signed(output_value), dut.saved_relu,
            accumulator_saturated, output_saturated
        );
        cycle = cycle + 1;
    end

    task automatic tick;
        @(posedge clk);
        #1;
    endtask

    task automatic run_example(input logic use_relu);
        @(negedge clk);
        start = 1;
        bias = 512;
        relu_enable = use_relu;
        input_valid = 0;
        input_last = 0;
        tick();

        @(negedge clk);
        start = 0;
        input_valid = 1;
        input_value = 64;
        weight_value = 32;
        tick();

        // This pair must be ignored because input_valid is low.
        @(negedge clk);
        input_valid = 0;
        input_last = 1;
        input_value = 2047;
        weight_value = 2047;
        tick();

        @(negedge clk);
        input_valid = 1;
        input_last = 0;
        input_value = 32;
        weight_value = -64;
        tick();

        @(negedge clk);
        input_last = 1;
        input_value = -64;
        weight_value = 16;
        tick();

        @(negedge clk);
        input_valid = 0;
        input_last = 0;
        tick();

        if (output_valid !== 1 || busy !== 0)
            $fatal(1, "Missing trace example output");
        if ($signed(output_value) !== (use_relu ? 0 : -8))
            $fatal(1, "Trace example output mismatch");
        if (accumulator_saturated !== 0 || output_saturated !== 0)
            $fatal(1, "Unexpected saturation");

        tick();
        if (output_valid !== 0)
            $fatal(1, "Completion pulse did not clear");
    endtask

    initial begin
        #10000;
        $fatal(1, "Trace simulation timed out");
    end

    initial begin
        trace_file = $fopen("build/sim/neuron-trace.csv", "w");
        if (trace_file == 0)
            $fatal(1, "Cannot create trace CSV");

        $fdisplay(trace_file,
            "cycle,reset,start,input_valid,input_last,busy,output_valid,input,weight,accumulator,output,relu,accumulator_saturated,output_saturated"
        );
        $dumpfile("build/sim/neuron-trace.vcd");
        $dumpvars(0, neuron_trace_tb);

        repeat (2) tick();
        @(negedge clk);
        reset = 0;
        tick();

        run_example(0);
        run_example(1);

        @(negedge clk);
        $fclose(trace_file);
        $display("PASS: trace examples produced -8 without ReLU and 0 with ReLU");
        $display("Saved build/sim/neuron-trace.csv and build/sim/neuron-trace.vcd");
        $finish;
    end
endmodule
