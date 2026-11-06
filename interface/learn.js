import { forward, backward, verifyExport } from "./network_math.js";
import { preprocessDrawing } from "./drawing_input.js";

const $ = id => document.getElementById(id);
const canvas = $("network");
const ctx = canvas.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let data, result, gradients;
let customSample = null;
const currentSample = () => $("input-mode").value === "drawing"
    ? customSample : data.samples[sampleIndex];
let nodes = [], layers = [], edges = [], projected = [];
let checkpointIndex = 0, sampleIndex = 0;
let selectedNeuron = { layer: 1, index: 0 };
let yaw = -0.28, pitch = 0.12, width = 0, height = 0;
let playing = false, phase = 0, lastTime = 0, frame = null, drag = null;
let targetYaw = -0.28, targetPitch = 0.12;
let cameraFrame = null, lastCameraTime = 0;
let displayedActivations = null, displayedGradients = null;
let visualTransition = null, visualFrame = null;
let chartProgress = 1, chartRevealStart = null, chartRevealFrame = null;
let chartMarkerEpoch = null, chartMarkerFrom = null, chartMarkerTo = null;
let chartMarkerStart = null, chartMarkerFrame = null;

const snapshot = () => data.snapshots[checkpointIndex];
const format = value => Math.abs(value) < 0.0001 && value !== 0
    ? value.toExponential(3) : value.toFixed(5);

function buildNodes() {
    nodes = [];
    layers = [];
    const dimensions = data.config.dimensions;

    dimensions.forEach((count, layer) => {
        const visibleCount = Math.min(count, layer === 0 ? 36 : 40);
        const group = [];

        for (let position = 0; position < visibleCount; position++) {
            const index = visibleCount === count ? position
                : Math.round(position * (count - 1) / (visibleCount - 1));
            const angle = position * 2.399963;
            const radius = Math.sqrt((position + 0.5) / visibleCount);
            const output = layer === dimensions.length - 1;
            const node = {
                layer, index,
                x: (layer - (dimensions.length - 1) / 2) * 205,
                y: output ? (position - (visibleCount - 1) / 2) * 25
                    : Math.sin(angle) * radius * 105,
                z: output ? 0 : Math.cos(angle) * radius * 95,
            };
            group.push(node);
            nodes.push(node);
        }
        layers.push(group);
    });

    $("display-count").textContent =
        `${nodes.length} of ${dimensions.reduce((a, b) => a + b, 0)} neurons shown`;
}

function buildEdges(previousWeights = new Map()) {
    edges = [];
    for (let layer = 0; layer < layers.length - 1; layer++) {
        for (const destination of layers[layer + 1]) {
            const incoming = layers[layer].map(source => {
                const key = `${layer}:${source.index}:${destination.index}`;
                const targetWeight =
                    snapshot().weights[layer][source.index][destination.index];
                const startWeight = previousWeights.has(key)
                    ? previousWeights.get(key) : targetWeight;
                return {
                    source, destination, key,
                    weight: startWeight, startWeight, targetWeight,
                };
            });
            incoming.sort((a, b) =>
                Math.abs(b.targetWeight) - Math.abs(a.targetWeight)
            );
            edges.push(...incoming.slice(0, 3));
        }
    }
}

function populateNeuronSelect() {
    $("neuron").replaceChildren();
    data.config.dimensions.forEach((count, layer) => {
        const group = document.createElement("optgroup");
        group.label = layer === 0 ? "Input pixels"
            : layer === data.config.dimensions.length - 1
                ? "Output neurons" : `Hidden layer ${layer}`;
        for (let index = 0; index < count; index++) {
            group.append(new Option(
                layer === 0 ? `Pixel ${index}` : `Neuron ${index}`,
                `${layer}:${index}`
            ));
        }
        $("neuron").append(group);
    });
    $("neuron").value = `${selectedNeuron.layer}:${selectedNeuron.index}`;
}

function inspectNeuron() {
    const { layer, index } = selectedNeuron;
    const values = [
        ["Layer / neuron", `${layer} / ${index}`],
        ["Activation", format(result.activations[layer][index])],
    ];
    if (layer > 0) {
        values.push(
            ["Weighted sum + bias", format(result.preactivations[layer - 1][index])],
            ["Bias", format(snapshot().biases[layer - 1][index])],
            ["∂ loss / ∂ weighted sum", gradients
                ? format(gradients[layer - 1][index]) : "Select a digit label"]
        );
    }
    if (layer === data.config.dimensions.length - 1) {
        values.push(["Softmax probability",
            `${(result.probabilities[index] * 100).toFixed(2)}%`]);
    }
    $("neuron-values").replaceChildren();
    for (const [label, value] of values) {
        const term = document.createElement("dt");
        const definition = document.createElement("dd");
        term.textContent = label;
        definition.textContent = value;
        $("neuron-values").append(term, definition);
    }
}

function drawInput() {
    const sample = currentSample();
    const resolution = data.config.resolution;
    const image = $("pixels");
    image.width = image.height = resolution;
    const context = image.getContext("2d");

    sample.pixels.forEach((value, index) => {
        const shade = Math.round((1 - value) * 255);
        context.fillStyle = `rgb(${shade},${shade},${shade})`;
        context.fillRect(index % resolution, Math.floor(index / resolution), 1, 1);
    });

    $("original").src = sample.original || `../${sample.path}`;
    $("original").alt = sample.original
        ? "Your submitted drawing" : `Original handwritten ${sample.label}`;
    $("sample-path").textContent = sample.original
        ? `Your drawing → ${resolution}×${resolution} input · processed in this browser`
        : `Original → ${resolution}×${resolution} input · ${sample.path}`;
}

function showPrediction() {
    const label = currentSample().label;
    const verdict = label === null ? ""
        : result.prediction === label ? " · correct" : ` · actual ${label}`;
    $("prediction").textContent = `Predicted ${result.prediction}${verdict}`;

    $("probabilities").replaceChildren();
    result.probabilities.forEach((probability, digit) => {
        const row = document.createElement("div");
        row.className = "probability-row";
        const name = document.createElement("span");
        name.textContent = digit;
        const track = document.createElement("div");
        track.className = "probability-track";
        const fill = document.createElement("div");
        fill.className = "probability-fill";
        fill.style.width = `${probability * 100}%`;
        track.append(fill);
        const value = document.createElement("span");
        value.textContent = `${(probability * 100).toFixed(1)}%`;
        row.append(name, track, value);
        $("probabilities").append(row);
    });
}

function resizeCanvas(element) {
    const box = element.getBoundingClientRect();
    const ratio = Math.min(devicePixelRatio || 1, 2);
    element.width = Math.round(box.width * ratio);
    element.height = Math.round(box.height * ratio);
    const context = element.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: box.width, height: box.height };
}

function drawChart(id, trainKey, validationKey, accuracy) {
    const { context, width: w, height: h } = resizeCanvas($(id));
    const history = data.history;
    const maximum = accuracy ? 1 : Math.max(
        ...history.map(row => Math.max(row[trainKey], row[validationKey]))
    ) * 1.05;
    const left = 48, right = w - 15, top = 12, bottom = h - 34;
    const finalEpoch = history.at(-1).epoch;
    const x = epoch => left + epoch / finalEpoch * (right - left);
    const y = value => bottom - value / maximum * (bottom - top);

    context.font = "11px sans-serif";
    for (let i = 0; i <= 4; i++) {
        const value = maximum * i / 4;
        context.strokeStyle = "#293548";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(left, y(value));
        context.lineTo(right, y(value));
        context.stroke();
        context.fillStyle = "#a7b5c8";
        context.fillText(
            accuracy ? `${Math.round(value * 100)}%` : value.toFixed(2),
            0, y(value) + 4
        );
    }

    const lastPoint = (history.length - 1) * chartProgress;
    const lastIndex = Math.floor(lastPoint);
    const partial = lastPoint - lastIndex;

    for (const [key, color] of [
        [trainKey, "#83d8c8"], [validationKey, "#e9b679"]
    ]) {
        context.strokeStyle = color;
        context.lineWidth = 1.8;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.beginPath();

        for (let index = 0; index <= lastIndex; index++) {
            const row = history[index];
            if (index === 0) context.moveTo(x(row.epoch), y(row[key]));
            else context.lineTo(x(row.epoch), y(row[key]));
        }

        if (lastIndex < history.length - 1 && partial > 0) {
            const from = history[lastIndex], to = history[lastIndex + 1];
            const epoch = from.epoch + (to.epoch - from.epoch) * partial;
            const value = from[key] + (to[key] - from[key]) * partial;
            context.lineTo(x(epoch), y(value));
        }
        context.stroke();
    }
    context.strokeStyle = "#e9eef5";
    context.lineWidth = 1;
    context.beginPath();
    const markerEpoch = chartMarkerEpoch ?? snapshot().epoch;
    context.moveTo(x(markerEpoch), top);
    context.lineTo(x(markerEpoch), bottom);
    context.stroke();
    context.fillStyle = "#a7b5c8";
    context.fillText("0", left, h - 12);
    context.fillText(`Epoch ${finalEpoch}`, Math.max(left, right - 80), h - 12);
}

function setChartMarker(epoch) {
    if (chartMarkerEpoch === null || reducedMotion) {
        chartMarkerEpoch = epoch;
        return;
    }

    chartMarkerFrom = chartMarkerEpoch;
    chartMarkerTo = epoch;
    chartMarkerStart = null;
    if (chartMarkerFrame === null) {
        chartMarkerFrame = requestAnimationFrame(animateChartMarker);
    }
}

function animateChartMarker(time) {
    if (chartMarkerStart === null) chartMarkerStart = time;
    const progress = Math.min((time - chartMarkerStart) / 360, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    chartMarkerEpoch = chartMarkerFrom
        + (chartMarkerTo - chartMarkerFrom) * eased;

    drawChart("loss-chart", "train_loss", "validation_loss", false);
    drawChart("accuracy-chart", "train_accuracy", "validation_accuracy", true);

    if (progress < 1) {
        chartMarkerFrame = requestAnimationFrame(animateChartMarker);
    } else {
        chartMarkerEpoch = chartMarkerTo;
        chartMarkerFrame = null;
        chartMarkerStart = null;
    }
}

function animateChartReveal(time) {
    if (chartRevealStart === null) chartRevealStart = time;
    const progress = Math.min((time - chartRevealStart) / 900, 1);
    chartProgress = 1 - Math.pow(1 - progress, 3);

    drawChart("loss-chart", "train_loss", "validation_loss", false);
    drawChart("accuracy-chart", "train_accuracy", "validation_accuracy", true);

    if (progress < 1) {
        chartRevealFrame = requestAnimationFrame(animateChartReveal);
    } else {
        chartProgress = 1;
        chartRevealFrame = null;
        chartRevealStart = null;
    }
}

function updateCalculation() {
    const sample = currentSample();
    const inspection = document.querySelector(".inspection");
    const previousEdgeWeights = new Map(
        edges.map(edge => [edge.key, edge.weight])
    );

    $("timeline").value = checkpointIndex;
    const metrics = data.history.find(row => row.epoch === snapshot().epoch);
    setChartMarker(snapshot().epoch);
    $("epoch").textContent =
        `Epoch ${snapshot().epoch} · train ${(metrics.train_accuracy * 100).toFixed(1)}% · validation ${(metrics.validation_accuracy * 100).toFixed(1)}%`;
    drawChart("loss-chart", "train_loss", "validation_loss", false);
    drawChart("accuracy-chart", "train_accuracy", "validation_accuracy", true);

    if (!sample) {
        result = null;
        gradients = null;
        displayedActivations = null;
        displayedGradients = null;
        visualTransition = null;
        if (visualFrame !== null) {
            cancelAnimationFrame(visualFrame);
            visualFrame = null;
        }
        edges = [];
        inspection.hidden = true;
        drawNetwork();
        $("phase").textContent = "Draw a digit, then select Predict drawing";
        return;
    }

    result = forward(sample.pixels, snapshot());
    gradients = sample.label === null
        ? null : backward(result, sample.label, snapshot());

    if ($("input-mode").value === "validation") {
        const reference = snapshot().probabilities[sampleIndex];
        const error = Math.max(
            ...result.probabilities.map((value, i) => Math.abs(value - reference[i]))
        );
        if (error > 1e-8) {
            throw new Error("Prediction differs from Python reference");
        }
    }

    document.querySelector(".replay .technical-note").textContent =
        "Values come from saved checkpoints. Moving signals illustrate calculation order, not hardware timing. "
        + (gradients
            ? "The backward view shows this drawing's loss gradients; replay does not update weights."
            : "Select a digit label to inspect loss gradients. This drawing uses forward calculations only.");

    inspection.hidden = false;
    buildEdges(previousEdgeWeights);
    beginVisualTransition();
    drawInput();
    showPrediction();
    inspectNeuron();
    drawNetwork();
}

function beginVisualTransition() {
    const targetActivations = result.activations.map(values => [...values]);
    const targetGradients = gradients
        ? gradients.map(values => [...values]) : null;

    if (!displayedActivations || reducedMotion) {
        displayedActivations = targetActivations;
        displayedGradients = targetGradients;
        edges.forEach(edge => { edge.weight = edge.targetWeight; });
        visualTransition = null;
        if (visualFrame !== null) {
            cancelAnimationFrame(visualFrame);
            visualFrame = null;
        }
        return;
    }

    const startActivations = displayedActivations.map(values => [...values]);
    const startGradients = targetGradients
        ? displayedGradients
            ? displayedGradients.map(values => [...values])
            : targetGradients.map(values => values.map(() => 0))
        : null;

    visualTransition = {
        startTime: null,
        startActivations,
        targetActivations,
        startGradients,
        targetGradients,
        startWeights: edges.map(edge => edge.startWeight),
        targetWeights: edges.map(edge => edge.targetWeight),
    };

    if (visualFrame === null) {
        visualFrame = requestAnimationFrame(animateVisualTransition);
    }
}

function animateVisualTransition(time) {
    const transition = visualTransition;
    if (!transition) {
        visualFrame = null;
        return;
    }

    if (transition.startTime === null) transition.startTime = time;
    const progress = Math.min((time - transition.startTime) / 520, 1);
    const eased = 1 - Math.pow(1 - progress, 3);

    displayedActivations = transition.startActivations.map((layer, index) =>
        layer.map((value, neuron) => value
            + (transition.targetActivations[index][neuron] - value) * eased)
    );

    displayedGradients = transition.targetGradients
        ? transition.startGradients.map((layer, index) =>
            layer.map((value, neuron) => value
                + (transition.targetGradients[index][neuron] - value) * eased)
        )
        : null;

    edges.forEach((edge, index) => {
        const start = transition.startWeights[index];
        const target = transition.targetWeights[index];
        edge.weight = start + (target - start) * eased;
    });

    drawNetwork();

    if (progress < 1) {
        visualFrame = requestAnimationFrame(animateVisualTransition);
        return;
    }

    displayedActivations = transition.targetActivations;
    displayedGradients = transition.targetGradients;
    edges.forEach(edge => { edge.weight = edge.targetWeight; });
    visualTransition = null;
    visualFrame = null;
    drawNetwork();
}

function project(node) {
    const x = node.x * Math.cos(yaw) + node.z * Math.sin(yaw);
    let z = -node.x * Math.sin(yaw) + node.z * Math.cos(yaw);
    const y = node.y * Math.cos(pitch) - z * Math.sin(pitch);
    z = node.y * Math.sin(pitch) + z * Math.cos(pitch);
    const perspective = 1000 / (1000 + z);
    const span = Math.max(800, (layers.length - 1) * 205 + 160);
    const scale = Math.min(width / span, height / 360);
    return {
        x: width / 2 + x * scale * perspective,
        y: height / 2 + y * scale * perspective,
        z, scale: perspective * Math.max(scale, 0.65),
    };
}

function drawNetwork() {
    ctx.clearRect(0, 0, width, height);
    if (!result) return;
    const shownActivations = displayedActivations ?? result.activations;
    const shownGradients = displayedGradients ?? gradients;
    const backwardPhase = phase >= 0.58 && gradients !== null;
    const progress = gradients === null ? phase
        : backwardPhase ? (phase - 0.58) / 0.42 : phase / 0.58;
    $("phase").textContent = backwardPhase
        ? "Backward view · single-drawing loss gradients ←"
        : "Forward view · neuron activations →";

    projected = nodes.map(node => ({ node, point: project(node) }));
    const positions = new Map(projected.map(item => [item.node, item.point]));
    const largestWeight = Math.max(0.001, ...edges.map(edge => Math.abs(edge.weight)));

    for (const edge of edges) {
        const a = positions.get(edge.source), b = positions.get(edge.destination);
        const strength = Math.abs(edge.weight) / largestWeight;
        ctx.strokeStyle = edge.weight >= 0 ? "#83d8c8" : "#e9b679";
        ctx.globalAlpha = 0.045 + strength * 0.22;
        ctx.lineWidth = 0.5 + strength;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (playing && !reducedMotion) {
        for (let i = 0; i < edges.length; i += 6) {
            const edge = edges[i];
            const stage = backwardPhase
                ? layers.length - 2 - edge.source.layer : edge.source.layer;
            const local = progress * (layers.length - 1) - stage;
            if (local < 0 || local > 1) continue;
            const t = backwardPhase ? 1 - local : local;
            const a = positions.get(edge.source), b = positions.get(edge.destination);
            ctx.fillStyle = backwardPhase ? "#e9b679" : "#b3f3e4";
            ctx.shadowColor = ctx.fillStyle;
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, 2.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    const maxima = layers.map((group, layer) => {
        const values = backwardPhase && layer > 0
            ? shownGradients[layer - 1] : shownActivations[layer];
        return Math.max(0.000001, ...values.map(Math.abs));
    });

    const sorted = [...projected].sort((a, b) => b.point.z - a.point.z);
    for (const { node, point } of sorted) {
        const value = backwardPhase && node.layer > 0
            ? shownGradients[node.layer - 1][node.index]
            : shownActivations[node.layer][node.index];
        const intensity = Math.sqrt(Math.abs(value) / maxima[node.layer]);
        const selected = node.layer === selectedNeuron.layer
            && node.index === selectedNeuron.index;
        const output = node.layer === layers.length - 1;
        const predicted = output && node.index === result.prediction;

        ctx.fillStyle = backwardPhase ? "#e9b679" : "#83d8c8";
        ctx.globalAlpha = 0.12 + intensity * 0.88;
        ctx.beginPath();
        ctx.arc(point.x, point.y, (output ? 4.5 : 2.8) * point.scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;

        if (selected || predicted) {
            ctx.shadowColor = predicted ? "#e9eef5" : "#83d8c8";
            ctx.shadowBlur = 12;
            ctx.strokeStyle = "#e9eef5";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(point.x, point.y, (output ? 8 : 6) * point.scale, 0, Math.PI * 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
        if (output) {
            ctx.fillStyle = predicted ? "#e9eef5" : "#a7b5c8";
            ctx.font = "12px sans-serif";
            ctx.fillText(node.index, point.x + 12, point.y + 4);
        }
    }

    ctx.textAlign = "center";
    ctx.font = "12px sans-serif";
    layers.forEach((group, layer) => {
        const point = project({ x: group[0].x, y: 145, z: 0 });
        const label = layer === 0 ? "Input"
            : layer === layers.length - 1 ? "Output" : `Hidden ${layer}`;
        ctx.fillStyle = "#a7b5c8";
        ctx.fillText(
            `${label} · ${data.config.dimensions[layer]}`,
            point.x, Math.min(height - 18, point.y)
        );
    });
    ctx.textAlign = "left";
}

function animateCamera(time) {
    const delta = lastCameraTime
        ? Math.min((time - lastCameraTime) / 1000, 0.05) : 1 / 60;
    lastCameraTime = time;
    const easing = 1 - Math.exp(-16 * delta);

    yaw += (targetYaw - yaw) * easing;
    pitch += (targetPitch - pitch) * easing;
    drawNetwork();

    if (Math.abs(targetYaw - yaw) + Math.abs(targetPitch - pitch) > 0.0003) {
        cameraFrame = requestAnimationFrame(animateCamera);
    } else {
        yaw = targetYaw;
        pitch = targetPitch;
        cameraFrame = null;
        lastCameraTime = 0;
        drawNetwork();
    }
}

function moveCamera() {
    if (reducedMotion) {
        yaw = targetYaw;
        pitch = targetPitch;
        drawNetwork();
    } else if (cameraFrame === null) {
        cameraFrame = requestAnimationFrame(animateCamera);
    }
}

function stopPlayback() {
    playing = false;
    $("play").textContent = "Play training replay";
}

function animate(time) {
    const delta = lastTime ? Math.min((time - lastTime) / 1000, 0.05) : 0;
    lastTime = time;
    if (playing) {
        phase += delta / 3.5;
        if (phase >= 1) {
            phase -= 1;
            if (checkpointIndex < data.snapshots.length - 1) {
                checkpointIndex++;
                updateCalculation();
            } else {
                phase = 0;
                stopPlayback();
            }
        }
    }
    drawNetwork();
    if (playing) frame = requestAnimationFrame(animate);
    else {
        frame = null;
        lastTime = 0;
    }
}

$("play").addEventListener("click", () => {
    if (playing) {
        stopPlayback();
        return;
    }
    if (checkpointIndex === data.snapshots.length - 1) {
        checkpointIndex = 0;
        phase = 0;
        updateCalculation();
    }
    playing = true;
    $("play").textContent = "Pause replay";
    if (frame === null) frame = requestAnimationFrame(animate);
});

$("timeline").addEventListener("input", () => {
    stopPlayback();
    checkpointIndex = Number($("timeline").value);
    phase = 0;
    updateCalculation();
});

$("sample").addEventListener("change", () => {
    sampleIndex = Number($("sample").value);
    updateCalculation();
});

$("camera").addEventListener("change", () => {
    const flat = $("camera").value === "flat";
    targetYaw = flat ? 0 : -0.28;
    targetPitch = flat ? 0 : 0.12;
    moveCamera();
});

$("neuron").addEventListener("change", () => {
    const [layer, index] = $("neuron").value.split(":").map(Number);
    selectedNeuron = { layer, index };
    inspectNeuron();
    drawNetwork();
});

canvas.addEventListener("pointerdown", event => {
    drag = { x: event.clientX, y: event.clientY, moved: false };
    canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", event => {
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
    targetYaw += dx * 0.005;
    targetPitch = Math.max(-0.6, Math.min(0.6, targetPitch + dy * 0.004));
    drag.x = event.clientX;
    drag.y = event.clientY;
    moveCamera();
});

canvas.addEventListener("pointerup", event => {
    if (drag && !drag.moved && result) {
        const box = canvas.getBoundingClientRect();
        const x = event.clientX - box.left, y = event.clientY - box.top;
        const nearest = [...projected].sort((a, b) =>
            Math.hypot(a.point.x - x, a.point.y - y)
            - Math.hypot(b.point.x - x, b.point.y - y)
        )[0];
        if (nearest && Math.hypot(nearest.point.x - x, nearest.point.y - y) < 16) {
            selectedNeuron = { layer: nearest.node.layer, index: nearest.node.index };
            $("neuron").value = `${selectedNeuron.layer}:${selectedNeuron.index}`;
            inspectNeuron();
            drawNetwork();
        }
    }
    drag = null;
});
canvas.addEventListener("pointercancel", () => { drag = null; });

function resize() {
    const sized = resizeCanvas(canvas);
    width = sized.width;
    height = sized.height;
    if (data && result) {
        drawNetwork();
        drawChart("loss-chart", "train_loss", "validation_loss", false);
        drawChart("accuracy-chart", "train_accuracy", "validation_accuracy", true);
    }
}

new ResizeObserver(resize).observe(canvas);
window.addEventListener("resize", resize);


function setupDrawing() {
    const drawing = $("draw-canvas");
    const pen = drawing.getContext("2d");
    let stroke = null;

    function message(value, error = false) {
        $("drawing-status").textContent = value;
        $("drawing-status").classList.toggle("error", error);
    }

    function resetCanvas() {
        pen.fillStyle = "white";
        pen.fillRect(0, 0, drawing.width, drawing.height);
        stroke = null;
    }

    function point(event) {
        const box = drawing.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(128,
                (event.clientX - box.left) * drawing.width / box.width)),
            y: Math.max(0, Math.min(128,
                (event.clientY - box.top) * drawing.height / box.height)),
        };
    }

    function drawSegment(from, to) {
        pen.strokeStyle = "black";
        pen.lineWidth = 7;
        pen.lineCap = "round";
        pen.lineJoin = "round";
        pen.beginPath();
        pen.moveTo(from.x, from.y);
        pen.lineTo(to.x, to.y);
        pen.stroke();
    }

    function label() {
        const value = $("drawing-label").value;
        return value === "" ? null : Number(value);
    }

    resetCanvas();

    $("input-mode").addEventListener("change", () => {
        if (!data) return;
        stopPlayback();
        phase = 0;
        const drawingMode = $("input-mode").value === "drawing";
        $("draw-panel").hidden = !drawingMode;
        $("sample-control").hidden = drawingMode;
        if (drawingMode) checkpointIndex = data.snapshots.length - 1;
        updateCalculation();
    });

    drawing.addEventListener("pointerdown", event => {
        if (!data || stroke || (event.pointerType === "mouse" && event.button !== 0)) {
            return;
        }
        event.preventDefault();
        stopPlayback();
        phase = 0;
        drawNetwork();
        const start = point(event);
        stroke = { ...start, pointerId: event.pointerId };
        drawing.setPointerCapture(event.pointerId);
        pen.fillStyle = "black";
        pen.beginPath();
        pen.arc(start.x, start.y, 3.5, 0, Math.PI * 2);
        pen.fill();
        message(customSample
            ? "Drawing changed. Select Predict drawing to update the displayed result."
            : "Select Predict drawing when you finish.");
    });

    drawing.addEventListener("pointermove", event => {
        if (!stroke || stroke.pointerId !== event.pointerId) return;
        event.preventDefault();
        const next = point(event);
        drawSegment(stroke, next);
        stroke = { ...next, pointerId: event.pointerId };
    });

    function finishStroke(event) {
        if (!stroke || stroke.pointerId !== event.pointerId) return;
        if (event.type === "pointerup") drawSegment(stroke, point(event));
        stroke = null;
        if (drawing.hasPointerCapture(event.pointerId)) {
            drawing.releasePointerCapture(event.pointerId);
        }
    }

    drawing.addEventListener("pointerup", finishStroke);
    drawing.addEventListener("pointercancel", finishStroke);
    drawing.addEventListener("lostpointercapture", () => { stroke = null; });

    $("predict-drawing").addEventListener("click", () => {
        if (!data || stroke) return;
        try {
            const prepared = preprocessDrawing(
                pen.getImageData(0, 0, drawing.width, drawing.height),
                data.config.resolution
            );
            customSample = {
                pixels: prepared.pixels,
                label: label(),
                original: drawing.toDataURL("image/png"),
            };
            stopPlayback();
            phase = 0;
            updateCalculation();
            message(
                `Predicted using epoch ${snapshot().epoch}. `
                + (prepared.edgeTouching ? "Ink touches the canvas edge. " : "")
                + "Move the checkpoint slider to compare saved models."
            );
        } catch (error) {
            message(error.message, true);
        }
    });

    $("clear-drawing").addEventListener("click", () => {
        resetCanvas();
        customSample = null;
        $("drawing-label").value = "";
        stopPlayback();
        phase = 0;
        if (data) updateCalculation();
        message("Draw a digit, then select Predict drawing.");
    });

    $("drawing-label").addEventListener("change", () => {
        if (!data || !customSample) return;
        customSample.label = label();
        updateCalculation();
    });
}

async function initialize() {
    try {
        const response = await fetch("../experiments/results/network-view.json");
        if (!response.ok) throw new Error(`Network export not found (${response.status})`);
        data = await response.json();
        const error = verifyExport(data);

        data.samples.forEach((sample, index) => {
            $("sample").add(new Option(
                `Digit ${sample.label} · ${sample.path.split("/").pop()}`, index
            ));
        });
        $("timeline").max = data.snapshots.length - 1;
        $("architecture").textContent =
            `${data.config.dimensions.join(" → ")} · ${data.config.parameters.toLocaleString()} parameters`;
        buildNodes();
        populateNeuronSelect();
        $("viewer").hidden = false;
        $("viewer").classList.add("is-ready");
        chartProgress = reducedMotion ? 1 : 0;
        resize();
        updateCalculation();
        if (!reducedMotion) {
            chartRevealFrame = requestAnimationFrame(animateChartReveal);
        }
        $("status").textContent =
            `Recorded weights loaded · all ${data.snapshots.length * data.samples.length} checkpoint predictions match Python · maximum difference ${error.toExponential(2)}`;
    } catch (error) {
        stopPlayback();
        $("viewer").hidden = true;
        $("status").classList.add("error");
        $("status").textContent =
            `${error.message}. Serve the repository root and run export_network.py.`;
        console.error(error);
    }
}

setupDrawing();
initialize();
