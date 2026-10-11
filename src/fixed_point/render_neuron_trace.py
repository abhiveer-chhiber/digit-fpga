"""Build a browser view of recorded neuron simulation values."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hardware neuron trace · Digit / FPGA</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap");
:root {
 color-scheme:dark;
 --ink:#e9eef5; --muted:#a7b5c8; --line:#293548;
 --teal:#83d8c8; --amber:#e9b679;
}
* { box-sizing:border-box }
body {
 margin:0; background:#090e16; color:var(--ink);
 font:14px/1.7 "Plus Jakarta Sans",sans-serif;
}
main { max-width:1180px; margin:auto; padding:48px 28px 64px }
h1 {
 font:400 clamp(40px,5vw,64px)/1.05 "Instrument Serif",Georgia,serif;
 letter-spacing:-.025em; margin:16px 0 20px;
}
h2 { font-size:16px; font-weight:600; margin:0 0 18px }
p { max-width:720px; color:var(--muted); margin:0 0 16px }
.eyebrow {
 font-size:10px; letter-spacing:.16em; text-transform:uppercase;
 color:var(--teal); margin-bottom:24px;
}
.controls {
 display:flex; flex-wrap:wrap; gap:8px; align-items:center;
 margin:30px 0 20px; padding-top:24px; border-top:1px solid var(--line);
}
button,input { font:inherit }
button {
 min-height:44px; padding:9px 18px; background:transparent;
 color:var(--ink); border:1px solid var(--line);
 border-radius:4px; cursor:pointer;
}
button:hover { background:#172131; border-color:#52657a }
#play { background:var(--teal); color:#102520; border-color:var(--teal) }
#play:hover { background:#b4e9de }
button:disabled { opacity:.35; cursor:default }
:focus-visible { outline:2px solid var(--teal); outline-offset:4px }
label { display:block; color:var(--muted); font-size:11px }
input[type=range] {
 width:100%; min-height:36px; accent-color:var(--teal); margin:4px 0 0;
}
output {
 margin-left:auto; color:var(--teal);
 font:12px ui-monospace,monospace;
}
.waveform {
 overflow-x:auto; background:#0b121e;
 border:1px solid var(--line); border-radius:6px;
 margin:18px 0; scrollbar-color:#40516a #0b121e;
}
canvas { display:block }
.details {
 display:grid; grid-template-columns:1.4fr 1fr; gap:56px;
 border-top:1px solid var(--line); padding-top:28px; margin-top:28px;
}
dl {
 display:grid; grid-template-columns:1fr 1fr;
 gap:0 22px; margin:0; font-size:12px;
}
dt,dd { border-bottom:1px solid #1d2939; padding:7px 0 }
dt { color:var(--muted) }
dd {
 margin:0; text-align:right; font-family:ui-monospace,monospace;
 font-variant-numeric:tabular-nums;
}
.note { font-size:11px; line-height:1.8 }
.cycle-heading { color:var(--teal) }
@media(max-width:650px) {
 main { padding:28px 18px 40px }
 .details { grid-template-columns:1fr; gap:30px }
 h1 { font-size:44px }
 .controls { gap:6px }
 button { padding:9px 13px }
 output { width:100%; margin:7px 0 0 }
}

.value-graph { margin:24px 0 18px }
.graph-heading {
 display:flex; align-items:baseline; justify-content:space-between;
 gap:16px; margin-bottom:8px;
}
.graph-heading h2 { margin:0 }
#value-chart { width:100%; height:290px }
.value-graph .note { margin-top:14px; max-width:800px }
.signal-details {
 border-top:1px solid var(--line);
 border-bottom:1px solid var(--line);
 padding:16px 0; margin:24px 0 16px;
}
.signal-details summary {
 cursor:pointer; color:var(--muted); font-size:12px;
}
.signal-details[open] summary { color:var(--ink) }
.signal-details .waveform { margin-bottom:4px }

</style>
</head>
<body>
<main>
<p class="note"><a href="overview.html" style="color:var(--teal);text-underline-offset:4px">← Project overview</a></p>
<p class="eyebrow">DIGIT / FPGA · RTL simulation</p>
<h1>Inside a hardware neuron.</h1>
<p>Recorded values from the SystemVerilog neuron, sampled just after each rising
clock edge. The same three input-weight pairs are tested first without ReLU,
then with ReLU. Expected outputs: −8 and 0 in the stored integer format.</p>
<p class="note">This is a short test example, not a handwriting prediction.
The clock period is a testbench setting; it is not a measured FPGA speed.</p>
<div class="controls">
<button id="previous" type="button">Previous</button>
<button id="play" type="button">Play trace</button>
<button id="next" type="button">Next</button>
<output id="cycle"></output>
</div>
<label for="timeline">Recorded clock cycle</label>
<input id="timeline" type="range" min="0" step="1" value="0">

<section class="value-graph">
 <div class="graph-heading">
  <h2>Accumulator</h2>
  <output id="accumulator-value"></output>
 </div>
 <canvas id="value-chart" role="img"
  aria-label="Recorded accumulator values across clock cycles"></canvas>
 <p class="note">Dots show recorded values after each clock edge.
  The smooth line connects those points as a visual guide; it does not
  show changes measured between clock edges. Values are divided by 4096.</p>
</section>
<details class="signal-details">
 <summary>Control signals and stored integers</summary>
<div class="waveform" tabindex="0" role="region" aria-label="Scrollable recorded signal chart">
<canvas id="chart" role="img" aria-label="Control signals and integer values across recorded clock cycles"></canvas>
</div>
</details>
<p class="note">Control signals show 0 or 1. Number rows show stored integers.
Input values are consumed only when the neuron is receiving data and input_valid
is high. The output is ready when output_valid is high.</p>
<section class="details">
<div><h2 id="selected-heading" class="cycle-heading">Selected cycle</h2><dl id="values"></dl></div>
<div><h2>Number scales</h2>
<p>Inputs, weights, and outputs are divided by 64 to recover their represented values.
The accumulator is divided by 4096. A stored output of −8 represents −0.125.</p>
<p class="note">The accumulator row shows the register before the final bias
addition. The completion cycle shows the finished output.</p>
</div>
</section>
</main>
<script>
const data = __TRACE_DATA__;
const signals = [
 ["Reset","reset",true],["Start","start",true],
 ["Input valid","input_valid",true],["Last input","input_last",true],
 ["Busy","busy",true],["Output valid","output_valid",true],
 ["Saved ReLU","relu",true],["Input","input",false],
 ["Weight","weight",false],["Accumulator","accumulator",false],
 ["Output","output",false],["Accumulator clipped","accumulator_saturated",true],
 ["Output clipped","output_saturated",true]
];
const $ = id => document.getElementById(id);
let selected = 0, timer = null;
const left = 170, step = 52, chartTop = 42, rowHeight = 30;
const width = left + data.length * step + 16;
const height = chartTop + signals.length * rowHeight + 16;
const canvas = $("chart"), ctx = canvas.getContext("2d");
const ratio = Math.min(devicePixelRatio || 1, 2);
canvas.width = Math.round(width * ratio);
canvas.height = Math.round(height * ratio);
canvas.style.width = width + "px";
canvas.style.height = height + "px";
ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
$("timeline").max = data.length - 1;


function draw() {
 ctx.clearRect(0,0,width,height);
 ctx.textBaseline = "middle";
 ctx.font = "11px ui-monospace,monospace";

 const selectedX = left + selected * step;
 ctx.fillStyle = "#122b2d";
 ctx.fillRect(selectedX,0,step,height);

 data.forEach((row,index) => {
  const x = left + index * step;
  ctx.strokeStyle = "#1d2939";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x,chartTop);
  ctx.lineTo(x,height);
  ctx.stroke();
  ctx.fillStyle = index === selected ? "#83d8c8" : "#77899e";
  ctx.textAlign = "center";
  ctx.fillText(row.cycle,x + step/2,22);
 });

 signals.forEach(([label,key,digital],index) => {
  const y = chartTop + index * rowHeight;
  ctx.strokeStyle = "#1d2939";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0,y + rowHeight);
  ctx.lineTo(width,y + rowHeight);
  ctx.stroke();

  ctx.textAlign = "left";
  ctx.fillStyle = "#a7b5c8";
  ctx.fillText(label,16,y + rowHeight/2);


  if (digital) {
   const color = key.includes("saturated") ? "#e9b679" : "#83d8c8";
   const high = y + 10, low = y + 22;

   ctx.lineCap = "round";
   ctx.lineJoin = "round";
   ctx.lineWidth = 1;
   ctx.strokeStyle = "#405566";
   ctx.beginPath();
   data.forEach((row,i) => {
    const x = left + i * step;
    const level = row[key] ? high : low;
    if (i === 0) ctx.moveTo(x,level);
    else ctx.lineTo(x,level);
    ctx.lineTo(x + step,level);
   });
   ctx.stroke();

   ctx.strokeStyle = color;
   ctx.lineWidth = 1.3;
   ctx.beginPath();
   data.forEach((row,i) => {
    if (!row[key]) return;
    const x = left + i * step;
    ctx.moveTo(x,high);
    ctx.lineTo(x + step,high);
   });
   ctx.stroke();

   const markerX = left + (selected + .5) * step;
   const markerY = data[selected][key] ? high : low;
   ctx.fillStyle = data[selected][key] ? color : "#a7b5c8";
   ctx.beginPath();
   ctx.arc(markerX,markerY,2.5,0,Math.PI*2);
   ctx.fill();
  } else {
   ctx.textAlign = "center";
   data.forEach((row,i) => {
    const active = key === "output" ? row.output_valid
      : key === "accumulator" ? row.busy
      : row.busy && row.input_valid;
    ctx.fillStyle = active ? "#e9eef5" : "#65758a";
    if (key === "output" && row.output_valid) {
     ctx.fillStyle = "#83d8c8";
    }
    ctx.fillText(row[key],left + i*step + step/2,y + rowHeight/2);
   });
  }
 });

 ctx.textAlign = "left";
 ctx.strokeStyle = "#83d8c8";
 ctx.lineWidth = 1;
 ctx.beginPath();
 ctx.moveTo(selectedX,0);
 ctx.lineTo(selectedX,height);
 ctx.stroke();
}



let cursorPosition = 0;
let cursorFrom = 0;
let cursorTarget = 0;
let cursorStarted = 0;
let cursorFrame = null;
const quietMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function moveCursor() {
 cursorFrom = cursorPosition;
 cursorTarget = selected;
 cursorStarted = performance.now();
 drawValueGraph();
}

function drawValueGraph() {

 const element = $("value-chart");
 const box = element.getBoundingClientRect();
 const w = box.width, h = box.height;
 if (!w || !h) return;
 const pixelRatio = Math.min(devicePixelRatio || 1,2);
 element.width = Math.round(w * pixelRatio);
 element.height = Math.round(h * pixelRatio);
 const context = element.getContext("2d");
 context.setTransform(pixelRatio,0,0,pixelRatio,0,0);

 const values = data.map(row => row.accumulator / 4096);
 const minimum = Math.min(0,...values);
 const maximum = Math.max(0,...values);
 const span = Math.max(maximum - minimum,.25);
 const lower = minimum - span * .18;
 const upper = maximum + span * .18;
 const marginLeft = 54, marginRight = w - 16;
 const marginTop = 18, marginBottom = h - 38;
 const x = index => marginLeft +
  index / Math.max(1,data.length-1) * (marginRight-marginLeft);
 const y = value => marginBottom -
  (value-lower)/(upper-lower) * (marginBottom-marginTop);

 context.font = "11px ui-monospace,monospace";
 context.textBaseline = "middle";
 for (let i=0;i<=4;i++) {
  const value = lower + (upper-lower) * i/4;
  const py = y(value);
  context.strokeStyle = "#293548";
  context.lineWidth = .6;
  context.beginPath();
  context.moveTo(marginLeft,py);
  context.lineTo(marginRight,py);
  context.stroke();
  context.fillStyle = "#a7b5c8";
  context.textAlign = "right";
  context.fillText(value.toFixed(2),marginLeft-10,py);
 }


 const elapsed = quietMotion ? 1 :
  Math.min(1,(performance.now()-cursorStarted)/360);
 const eased = elapsed*elapsed*(3-2*elapsed);
 cursorPosition = cursorFrom + (cursorTarget-cursorFrom)*eased;

 const segment = Math.min(Math.floor(cursorPosition),values.length-2);
 const t = cursorPosition-segment;
 const curveX = 1.5*t-1.5*t*t+t*t*t;
 const curveY = t*t*(3-2*t);
 const currentX = x(segment)+(x(segment+1)-x(segment))*curveX;
 const currentValue = values[segment]+
  (values[segment+1]-values[segment])*curveY;

 context.strokeStyle = "#405b62";
 context.lineWidth = 1;
 context.setLineDash([3,5]);
 context.beginPath();
 context.moveTo(currentX,marginTop);
 context.lineTo(currentX,marginBottom);
 context.stroke();
 context.setLineDash([]);

 context.strokeStyle = "#83d8c8";
 context.lineWidth = 2;
 context.lineCap = "round";
 context.lineJoin = "round";
 context.beginPath();
 context.moveTo(x(0),y(values[0]));
 for (let i=1;i<values.length;i++) {
  const midpoint = (x(i-1)+x(i))/2;
  context.bezierCurveTo(
   midpoint,y(values[i-1]),
   midpoint,y(values[i]),
   x(i),y(values[i])
  );
 }
 context.stroke();

 values.forEach((value,index) => {
  context.fillStyle = "#090e16";
  context.strokeStyle = index === selected ? "#e9eef5" : "#83d8c8";
  context.lineWidth = index === selected ? 2 : 1.3;
  context.beginPath();
  context.arc(x(index),y(value),index === selected ? 5 : 2.5,0,Math.PI*2);
  context.fill();
  context.stroke();
 });

 context.fillStyle = "#a7b5c8";
 context.textBaseline = "alphabetic";
 context.textAlign = "left";
 context.fillText("Cycle 0",marginLeft,h-10);
 context.textAlign = "right";
 context.fillText("Cycle "+data.at(-1).cycle,marginRight,h-10);
 context.fillStyle = "#e9eef5";
 context.beginPath();
 context.arc(currentX,y(currentValue),4,0,Math.PI*2);
 context.fill();

 if (elapsed < 1 && cursorFrame === null) {
  cursorFrame = requestAnimationFrame(() => {
   cursorFrame = null;
   drawValueGraph();
  });
 }

 $("accumulator-value").textContent =
  values[selected].toFixed(4)+" · stored "+data[selected].accumulator;
}

function update() {
 const row=data[selected];
 $("timeline").value=selected;

 $("cycle").textContent = "Cycle " + row.cycle + " / " + data.at(-1).cycle;
 const state = row.reset ? "Reset"
   : row.output_valid ? "Output ready"
   : row.start ? "Start"
   : row.busy && row.input_valid ? "Receive input"
   : row.busy ? "Busy"
   : "Idle";
 $("selected-heading").textContent = "Cycle " + row.cycle + " · " + state;

 $("previous").disabled=selected===0;
 $("next").disabled=selected===data.length-1;
 $("values").replaceChildren();
 signals.forEach(([label,key]) => {
  const dt=document.createElement("dt"),dd=document.createElement("dd");

  dt.textContent=label;
  dd.textContent=row[key];
  if(row[key] !== 0) dd.style.color="#83d8c8";

  $("values").append(dt,dd);
 });
 draw();
 moveCursor();
}
function stop() {
 if(timer!==null)clearInterval(timer);
 timer=null;$("play").textContent="Play trace";
}
$("timeline").addEventListener("input",()=>{stop();selected=Number($("timeline").value);update();});
$("previous").addEventListener("click",()=>{stop();selected=Math.max(0,selected-1);update();});
$("next").addEventListener("click",()=>{stop();selected=Math.min(data.length-1,selected+1);update();});
$("play").addEventListener("click",()=>{
 if(timer!==null){stop();return;}
 if(selected===data.length-1)selected=0;
 update();$("play").textContent="Pause";
 timer=setInterval(()=>{
  selected++;update();
  if(selected===data.length-1)stop();
 },450);
});
document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
new ResizeObserver(drawValueGraph).observe($("value-chart"));
update();
</script>
</body>
</html>
"""


def render_trace(source):
    with source.open(newline="") as handle:
        rows = [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows or [row["cycle"] for row in rows] != list(range(len(rows))):
        raise ValueError("Expected consecutive recorded cycles")
    fields = {
        "cycle", "reset", "start", "input_valid", "input_last",
        "busy", "output_valid", "input", "weight", "accumulator",
        "output", "relu", "accumulator_saturated", "output_saturated",
    }
    if any(set(row) != fields for row in rows):
        raise ValueError("Unexpected trace fields")
    completed = [row for row in rows if row["output_valid"]]
    if [(row["output"], row["relu"]) for row in completed] != [(-8, 0), (0, 1)]:
        raise ValueError("Trace does not match the expected examples")
    return PAGE.replace("__TRACE_DATA__", json.dumps(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="build/sim/neuron-trace.csv")
    parser.add_argument("--output", default="build/sim/neuron-trace.html")
    args = parser.parse_args()
    source = ROOT / args.input
    destination = ROOT / args.output
    html = render_trace(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"Saved {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
