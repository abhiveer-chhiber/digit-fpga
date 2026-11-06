import { forward, backward, verifyExport } from "./network_math.js";
import { preprocessDrawing } from "./drawing_input.js";

const $ = id => document.getElementById(id);
const canvas = $("network");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let data, result, gradients;
let customSample = null;
const currentSample = () => $("input-mode").value === "drawing"
    ? customSample : data.samples[sampleIndex];
let nodes = [], layers = [], edges = [], projected = [];
let networkView = null;
let weightRevision = 0;
let checkpointIndex = 0, sampleIndex = 0;
let selectedNeuron = { layer: 1, index: 0 };
let yaw = -0.28, pitch = 0.12;
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

async function createNetworkScene(canvas) {
    const THREE = await import(
        "https://cdn.jsdelivr.net/npm/three@0.186.0/build/three.module.js"
    );

    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: false,
        alpha: false,
        powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.35));
    renderer.setClearColor(0x000000, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 4000);
    const network = new THREE.Group();
    scene.add(network);

    const layerColors = [
        new THREE.Color("#FF5A5F"),
        new THREE.Color("#2EE6D6"),
        new THREE.Color("#4DA3FF"),
        new THREE.Color("#9B7BFF"),
    ];
    const positiveColor = new THREE.Color("#7CFF6B");
    const negativeColor = new THREE.Color("#9B7BFF");

    let width = 1;
    let height = 1;
    let nodes = [];
    let edges = [];
    let layers = [];
    let layerCenters = [];
    let lastActivations = null;
    let lastGradients = null;
    let lastBackwardPhase = null;
    let lastWeightRevision = -1;
    let lastSelectedKey = "";
    let lastPrediction = null;
    let lastMarkerKey = "";
    let lastWasPlaying = false;
    let labelsDirty = true;
    let currentYaw = Number.NaN;
    let currentPitch = Number.NaN;

    let nodeGeometry = new THREE.BufferGeometry();
    const nodeMaterial = new THREE.ShaderMaterial({
        uniforms: { uPixelRatio: { value: renderer.getPixelRatio() } },
        vertexShader: `
            attribute vec3 color;
            attribute float aSize;
            uniform float uPixelRatio;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = aSize * uPixelRatio;
                gl_Position = projectionMatrix * viewPosition;
            }
        `,
        fragmentShader: `
            varying vec3 vColor;
            void main() {
                float distanceFromCenter = length(gl_PointCoord - vec2(0.5));
                float core = 1.0 - smoothstep(0.12, 0.24, distanceFromCenter);
                float halo = 0.24 * (1.0 - smoothstep(0.20, 0.50, distanceFromCenter));
                float alpha = max(core, halo);
                if (alpha < 0.01) discard;
                gl_FragColor = vec4(vColor * (0.82 + 0.38 * core), alpha);
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }
        `,
        transparent: true,
        depthWrite: false,
        depthTest: true,
        blending: THREE.AdditiveBlending,
    });
    const nodePoints = new THREE.Points(nodeGeometry, nodeMaterial);
    nodePoints.frustumCulled = false;

    let edgeGeometry = new THREE.BufferGeometry();
    const edgeMaterial = new THREE.ShaderMaterial({
        uniforms: { uOpacity: { value: 0.78 } },
        vertexShader: `
            attribute vec3 color;
            attribute float aAlpha;
            varying vec3 vColor;
            varying float vAlpha;
            void main() {
                vColor = color;
                vAlpha = aAlpha;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform float uOpacity;
            varying vec3 vColor;
            varying float vAlpha;
            void main() {
                gl_FragColor = vec4(vColor, vAlpha * uOpacity);
            }
        `,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
    });
    const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    edgeLines.frustumCulled = false;

    const markerGeometry = new THREE.BufferGeometry();
    markerGeometry.setAttribute(
        "position", new THREE.BufferAttribute(new Float32Array(6), 3)
    );
    markerGeometry.setDrawRange(0, 0);
    const markerMaterial = new THREE.ShaderMaterial({
        uniforms: {
            uPixelRatio: { value: renderer.getPixelRatio() },
            uColor: { value: new THREE.Color("#F2F2F2") },
        },
        vertexShader: `
            uniform float uPixelRatio;
            void main() {
                gl_PointSize = 13.0 * uPixelRatio;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform vec3 uColor;
            void main() {
                float distanceFromCenter = length(gl_PointCoord - vec2(0.5));
                float ring = 1.0 - smoothstep(0.025, 0.065, abs(distanceFromCenter - 0.34));
                float halo = 0.16 * (1.0 - smoothstep(0.20, 0.50, distanceFromCenter));
                float alpha = max(ring, halo);
                if (alpha < 0.01) discard;
                gl_FragColor = vec4(uColor, alpha * 0.9);
                #include <tonemapping_fragment>
                #include <colorspace_fragment>
            }
        `,
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
    });
    const markers = new THREE.Points(markerGeometry, markerMaterial);
    markers.frustumCulled = false;
    markers.renderOrder = 1;
    nodePoints.renderOrder = 2;

    const signalGeometry = new THREE.BufferGeometry();
    const maxSignals = 360;
    signalGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(maxSignals * 3), 3)
    );
    signalGeometry.setDrawRange(0, 0);
    const signalMaterial = new THREE.PointsMaterial({
        color: "#7CFF6B",
        size: 3.2,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.82,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
    });
    const signals = new THREE.Points(signalGeometry, signalMaterial);
    signals.frustumCulled = false;
    signals.renderOrder = 3;

    network.add(edgeLines, markers, nodePoints, signals);

    const labelLayer = document.createElement("div");
    labelLayer.className = "network-layer-labels";
    canvas.parentElement.append(labelLayer);
    let labelElements = [];

    function resize() {
        const box = canvas.getBoundingClientRect();
        width = Math.max(1, box.width);
        height = Math.max(1, box.height);
        renderer.setSize(width, height, false);
        camera.aspect = width / height;

        const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
        const fitHeight = 155 / Math.tan(halfFov);
        const fitWidth = 360 / (camera.aspect * Math.tan(halfFov));
        camera.position.set(0, 0, Math.max(fitHeight, fitWidth) * 1.18);
        camera.lookAt(0, 0, 0);
        camera.updateProjectionMatrix();
        labelsDirty = true;
    }

    function setRotation(yaw, pitch) {
        if (yaw === currentYaw && pitch === currentPitch) return;
        currentYaw = yaw;
        currentPitch = pitch;
        network.rotation.set(pitch, yaw, 0);
        network.updateMatrixWorld(true);
        labelsDirty = true;
    }

    function setNodes(nextNodes) {
        nodes = nextNodes;
        const positions = new Float32Array(nodes.length * 3);
        const colors = new Float32Array(nodes.length * 3);
        const sizes = new Float32Array(nodes.length);

        nodes.forEach((node, index) => {
            const offset = index * 3;
            positions[offset] = node.x;
            positions[offset + 1] = node.y;
            positions[offset + 2] = node.z;
            const base = layerColors[node.layer] || layerColors[0];
            colors[offset] = base.r * 0.55;
            colors[offset + 1] = base.g * 0.55;
            colors[offset + 2] = base.b * 0.55;
            sizes[index] = 3.5;
        });

        nodeGeometry.dispose();
        nodeGeometry = new THREE.BufferGeometry();
        nodeGeometry.setAttribute(
            "position", new THREE.BufferAttribute(positions, 3)
        );
        nodeGeometry.setAttribute(
            "color", new THREE.BufferAttribute(colors, 3)
        );
        nodeGeometry.setAttribute(
            "aSize", new THREE.BufferAttribute(sizes, 1)
        );
        nodePoints.geometry = nodeGeometry;

        layers = [];
        nodes.forEach(node => {
            if (!layers[node.layer]) layers[node.layer] = [];
            layers[node.layer].push(node);
        });
        layerCenters = layers.map(group => group.reduce((center, node) => ({
            x: center.x + node.x / group.length,
            y: center.y + node.y / group.length,
            z: center.z + node.z / group.length,
        }), { x: 0, y: 0, z: 0 }));
        labelsDirty = true;
        lastActivations = null;
        lastGradients = null;
        lastBackwardPhase = null;

        labelLayer.replaceChildren();
        labelElements = layers.map((group, layer) => {
            const label = document.createElement("span");
            label.className = "network-layer-label";
            label.style.color = (layerColors[layer] || layerColors[0]).getStyle();
            label.textContent = layer === 0 ? "INPUT"
                : layer === layers.length - 1 ? "OUTPUT" : `HIDDEN ${layer}`;
            labelLayer.append(label);
            return label;
        });
    }

    function setEdges(nextEdges) {
        edges = nextEdges;
        const positions = new Float32Array(edges.length * 6);
        const colors = new Float32Array(edges.length * 6);
        const alphas = new Float32Array(edges.length * 2);

        edges.forEach((edge, index) => {
            const offset = index * 6;
            positions[offset] = edge.source.x;
            positions[offset + 1] = edge.source.y;
            positions[offset + 2] = edge.source.z;
            positions[offset + 3] = edge.destination.x;
            positions[offset + 4] = edge.destination.y;
            positions[offset + 5] = edge.destination.z;
        });

        edgeGeometry.dispose();
        edgeGeometry = new THREE.BufferGeometry();
        edgeGeometry.setAttribute(
            "position", new THREE.BufferAttribute(positions, 3)
        );
        edgeGeometry.setAttribute(
            "color", new THREE.BufferAttribute(colors, 3)
        );
        edgeGeometry.setAttribute(
            "aAlpha", new THREE.BufferAttribute(alphas, 1)
        );
        edgeGeometry.setDrawRange(0, edges.length * 2);
        edgeLines.geometry = edgeGeometry;
        lastWeightRevision = -1;
        lastSelectedKey = "";
        lastPrediction = null;
    }

    function projectPoint(point) {
        const projected = new THREE.Vector3(
            point.x, point.y, point.z
        ).applyMatrix4(network.matrixWorld).project(camera);
        return {
            x: (projected.x * 0.5 + 0.5) * width,
            y: (1 - (projected.y * 0.5 + 0.5)) * height,
            z: projected.z,
        };
    }

    function projectNodes() {
        network.updateMatrixWorld(true);
        return nodes.map(node => ({ node, point: projectPoint(node) }));
    }

    function render(state) {
        const activationChanged = state.activations !== lastActivations
            || state.gradients !== lastGradients
            || state.backwardPhase !== lastBackwardPhase;

        if (activationChanged) {
            const maxima = layers.map((group, layer) => {
                const values = state.backwardPhase && layer > 0
                    ? state.gradients[layer - 1] : state.activations[layer];
                return Math.max(0.000001, ...values.map(Math.abs));
            });
            const nodeColors = nodeGeometry.attributes.color.array;
            const nodeSizes = nodeGeometry.attributes.aSize.array;

            nodes.forEach((node, index) => {
                const values = state.backwardPhase && node.layer > 0
                    ? state.gradients[node.layer - 1] : state.activations[node.layer];
                const intensity = Math.sqrt(
                    Math.abs(values[node.index]) / maxima[node.layer]
                );
                const brightness = 0.20 + intensity * 0.80;
                const base = layerColors[node.layer] || layerColors[0];
                const offset = index * 3;
                nodeColors[offset] = base.r * brightness;
                nodeColors[offset + 1] = base.g * brightness;
                nodeColors[offset + 2] = base.b * brightness;
                nodeSizes[index] = 3.2 + intensity * 1.8;
            });

            nodeGeometry.attributes.color.needsUpdate = true;
            nodeGeometry.attributes.aSize.needsUpdate = true;
            lastActivations = state.activations;
            lastGradients = state.gradients;
            lastBackwardPhase = state.backwardPhase;
        }

        const selectedKey = `${state.selectedNeuron.layer}:${state.selectedNeuron.index}`;
        const connectionsChanged = state.weightRevision !== lastWeightRevision
            || selectedKey !== lastSelectedKey
            || state.prediction !== lastPrediction;

        if (connectionsChanged) {
            let largestWeight = 0.001;
            for (const edge of edges) {
                largestWeight = Math.max(largestWeight, Math.abs(edge.weight));
            }

            const edgeColors = edgeGeometry.attributes.color.array;
            const edgeAlphas = edgeGeometry.attributes.aAlpha.array;
            edges.forEach((edge, index) => {
                const color = edge.weight >= 0 ? positiveColor : negativeColor;
                const strength = Math.sqrt(Math.abs(edge.weight) / largestWeight);
                const touchesSelected =
                    (edge.source.layer === state.selectedNeuron.layer
                        && edge.source.index === state.selectedNeuron.index)
                    || (edge.destination.layer === state.selectedNeuron.layer
                        && edge.destination.index === state.selectedNeuron.index);
                const leadsToPrediction =
                    edge.destination.layer === layers.length - 1
                    && edge.destination.index === state.prediction;
                const highlighted = touchesSelected || leadsToPrediction;
                const alpha = highlighted
                    ? 0.075 + strength * 0.14
                    : 0.0007 + strength * 0.0038;
                const offset = index * 6;
                edgeColors[offset] = color.r;
                edgeColors[offset + 1] = color.g;
                edgeColors[offset + 2] = color.b;
                edgeColors[offset + 3] = color.r;
                edgeColors[offset + 4] = color.g;
                edgeColors[offset + 5] = color.b;
                edgeAlphas[index * 2] = alpha;
                edgeAlphas[index * 2 + 1] = alpha;
            });
            edgeGeometry.attributes.color.needsUpdate = true;
            edgeGeometry.attributes.aAlpha.needsUpdate = true;
            lastWeightRevision = state.weightRevision;
            lastSelectedKey = selectedKey;
            lastPrediction = state.prediction;
        }

        const markerKey = `${selectedKey}|${state.prediction}`;
        if (markerKey !== lastMarkerKey) {
            const markerPositions = markerGeometry.attributes.position.array;
            let markerCount = 0;
            const selected = nodes.find(node =>
                node.layer === state.selectedNeuron.layer
                && node.index === state.selectedNeuron.index
            );
            const predicted = nodes.find(node =>
                node.layer === layers.length - 1 && node.index === state.prediction
            );
            for (const node of [selected, predicted]) {
                if (!node || (markerCount && node === selected && selected === predicted)) {
                    continue;
                }
                const offset = markerCount * 3;
                markerPositions[offset] = node.x;
                markerPositions[offset + 1] = node.y;
                markerPositions[offset + 2] = node.z;
                markerCount++;
            }
            markerGeometry.setDrawRange(0, markerCount);
            markerGeometry.attributes.position.needsUpdate = true;
            lastMarkerKey = markerKey;
        }

        if (state.playing) {
            const signalPositions = signalGeometry.attributes.position.array;
            let signalCount = 0;
            const stride = Math.max(1, Math.ceil(edges.length / maxSignals));
            for (let index = 0; index < edges.length; index += stride) {
                const edge = edges[index];
                const stage = state.backwardPhase
                    ? layers.length - 2 - edge.source.layer
                    : edge.source.layer;
                const local = state.progress * (layers.length - 1) - stage;
                if (local < 0 || local > 1) continue;

                const amount = state.backwardPhase ? 1 - local : local;
                const inverse = 1 - amount;
                const offset = signalCount * 3;
                signalPositions[offset] =
                    inverse * edge.source.x + amount * edge.destination.x;
                signalPositions[offset + 1] =
                    inverse * edge.source.y + amount * edge.destination.y;
                signalPositions[offset + 2] =
                    inverse * edge.source.z + amount * edge.destination.z;
                signalCount++;
            }
            signalGeometry.setDrawRange(0, signalCount);
            signalGeometry.attributes.position.needsUpdate = true;
            signalMaterial.color.set(state.backwardPhase ? "#9B7BFF" : "#7CFF6B");
        } else if (lastWasPlaying) {
            signalGeometry.setDrawRange(0, 0);
        }
        lastWasPlaying = state.playing;

        if (labelsDirty) {
            network.updateMatrixWorld(true);
            labelElements.forEach((label, layer) => {
                const center = layerCenters[layer];
                const point = projectPoint({ ...center, y: 145 });
                label.style.left = `${point.x}px`;
                label.style.top = `${Math.min(height - 18, Math.max(22, point.y))}px`;
                label.textContent = `${layer === 0 ? "INPUT"
                    : layer === layers.length - 1 ? "OUTPUT" : `HIDDEN ${layer}`} · ${state.dimensions[layer]}`;
            });
            labelsDirty = false;
        }

        renderer.render(scene, camera);
    }

    resize();

    return {
        resize,
        setNodes,
        setEdges,
        setRotation,
        projectNodes,
        render,
    };
}

function buildNodes() {
    nodes = [];
    layers = [];

    const dimensions = data.config.dimensions;
    const goldenAngle = 2.399963;
    const shapes = [
        { x: -265, y: 4, z: -12, rx: 58, ry: 98, rz: 55 },
        { x: -88, y: -18, z: 14, rx: 94, ry: 128, rz: 78 },
        { x: 94, y: 15, z: -12, rx: 83, ry: 112, rz: 70 },
        { x: 265, y: 0, z: 5, rx: 48, ry: 60, rz: 42 },
    ];

    dimensions.forEach((count, layer) => {
        const shape = shapes[layer];
        const group = [];

        for (let position = 0; position < count; position++) {
            const vertical = 1 - 2 * (position + 0.5) / count;
            const ring = Math.sqrt(1 - vertical * vertical);
            const angle = position * goldenAngle;
            const node = {
                layer,
                index: position,
                x: shape.x
                    + Math.cos(angle) * ring * shape.rx
                    + Math.sin(angle * 2.4 + layer) * 7,
                y: shape.y
                    + vertical * shape.ry
                    + Math.sin(angle * 1.3 + layer) * 8,
                z: shape.z
                    + Math.sin(angle) * ring * shape.rz
                    + Math.cos(angle * 1.8 + layer) * 8,
            };
            group.push(node);
            nodes.push(node);
        }

        layers.push(group);
    });

    networkView.setNodes(nodes);
    $("display-count").textContent = `${nodes.length} neurons`;
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
                    source,
                    destination,
                    key,
                    weight: startWeight,
                    startWeight,
                    targetWeight,
                };
            });
            edges.push(...incoming);
        }
    }

    networkView.setEdges(edges);
    weightRevision++;
    $("display-count").textContent =
        `${nodes.length} neurons · ${edges.length.toLocaleString()} weights`;
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
    weightRevision++;
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
    weightRevision++;

    drawNetwork();

    if (progress < 1) {
        visualFrame = requestAnimationFrame(animateVisualTransition);
        return;
    }

    displayedActivations = transition.targetActivations;
    displayedGradients = transition.targetGradients;
    edges.forEach(edge => { edge.weight = edge.targetWeight; });
    weightRevision++;
    visualTransition = null;
    visualFrame = null;
    drawNetwork();
}

function drawNetwork() {
    if (!result || !networkView) return;

    const shownActivations = displayedActivations ?? result.activations;
    const shownGradients = displayedGradients ?? gradients;
    const backwardPhase = phase >= 0.58 && gradients !== null;
    const progress = gradients === null ? phase
        : backwardPhase ? (phase - 0.58) / 0.42 : phase / 0.58;

    $("phase").textContent = backwardPhase
        ? "Backward view · single-drawing loss gradients ←"
        : "Forward view · neuron activations →";

    networkView.setRotation(yaw, pitch);
    projected = networkView.projectNodes();
    networkView.render({
        activations: shownActivations,
        gradients: shownGradients,
        backwardPhase,
        selectedNeuron,
        prediction: result.prediction,
        weightRevision,
        playing,
        progress,
        dimensions: data.config.dimensions,
    });
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
    if (networkView) networkView.resize();
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
        networkView = await createNetworkScene(canvas);
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
