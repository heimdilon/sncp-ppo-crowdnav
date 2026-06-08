const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const world = { min: -6, max: 6 };
const robotRadius = 0.3;
const humanRadius = 0.3;
const robotMaxSpeed = 0.26;

let selectedHuman = 0;
let dragTarget = null;

let state = {
  name: "custom_crossing",
  time_step: 0.25,
  max_time: 50,
  human_motion_model: "linear",
  human_dodge_robot: false,
  robot: {
    position: { x: -4, y: 0 },
    goal: { x: 4, y: 0 },
    theta_deg: 0
  },
  humans: [
    { id: "h1", position: { x: 0, y: -2.0 }, theta_deg: 90, speed: 0.2, goal: { x: 0, y: 4 } },
    { id: "h2", position: { x: 0, y: 2.0 }, theta_deg: -90, speed: 0.2, goal: { x: 0, y: -4 } },
    { id: "h3", position: { x: -1.5, y: -1.5 }, theta_deg: 45, speed: 0.18, goal: { x: 3, y: 3 } },
    { id: "h4", position: { x: 1.5, y: 1.5 }, theta_deg: -135, speed: 0.18, goal: { x: -3, y: -3 } },
    { id: "h5", position: { x: -1.5, y: 1.5 }, theta_deg: -45, speed: 0.18, goal: { x: 3, y: -3 } }
  ]
};

const inputs = {
  scenarioName: document.getElementById("scenarioName"),
  motionModel: document.getElementById("motionModel"),
  maxTime: document.getElementById("maxTime"),
  timeStep: document.getElementById("timeStep"),
  robotX: document.getElementById("robotX"),
  robotY: document.getElementById("robotY"),
  robotTheta: document.getElementById("robotTheta"),
  goalX: document.getElementById("goalX"),
  goalY: document.getElementById("goalY")
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toScreen(point) {
  const span = world.max - world.min;
  return {
    x: ((point.x - world.min) / span) * canvas.width,
    y: canvas.height - ((point.y - world.min) / span) * canvas.height
  };
}

function toWorld(point) {
  const span = world.max - world.min;
  return {
    x: clamp(world.min + (point.x / canvas.width) * span, world.min, world.max),
    y: clamp(world.min + ((canvas.height - point.y) / canvas.height) * span, world.min, world.max)
  };
}

function metersToPixels(meters) {
  return (meters / (world.max - world.min)) * canvas.width;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function readMainInputs() {
  state.name = inputs.scenarioName.value || "custom_scenario";
  state.human_motion_model = inputs.motionModel.value;
  state.max_time = Number(inputs.maxTime.value) || 50;
  state.time_step = Number(inputs.timeStep.value) || 0.25;
  state.robot.position.x = Number(inputs.robotX.value) || 0;
  state.robot.position.y = Number(inputs.robotY.value) || 0;
  state.robot.theta_deg = Number(inputs.robotTheta.value) || 0;
  state.robot.goal.x = Number(inputs.goalX.value) || 0;
  state.robot.goal.y = Number(inputs.goalY.value) || 0;
}

function syncMainInputs() {
  inputs.scenarioName.value = state.name;
  inputs.motionModel.value = state.human_motion_model;
  inputs.maxTime.value = state.max_time;
  inputs.timeStep.value = state.time_step;
  inputs.robotX.value = state.robot.position.x;
  inputs.robotY.value = state.robot.position.y;
  inputs.robotTheta.value = state.robot.theta_deg;
  inputs.goalX.value = state.robot.goal.x;
  inputs.goalY.value = state.robot.goal.y;
}

function updateHumanGoalFromHeading(human) {
  const radians = (human.theta_deg * Math.PI) / 180;
  human.goal = {
    x: round1(human.position.x + 6 * Math.cos(radians)),
    y: round1(human.position.y + 6 * Math.sin(radians))
  };
}

function renderHumanTable() {
  const body = document.getElementById("humanTableBody");
  body.innerHTML = "";
  state.humans.forEach((human, index) => {
    const row = document.createElement("tr");
    if (index === selectedHuman) row.classList.add("selected");
    row.addEventListener("click", () => {
      selectedHuman = index;
      draw();
    });

    const fields = [
      ["id", "text"],
      ["position.x", "number"],
      ["position.y", "number"],
      ["theta_deg", "number"],
      ["speed", "number"],
      ["goal.x", "number"],
      ["goal.y", "number"]
    ];

    fields.forEach(([field, type]) => {
      const cell = document.createElement("td");
      const input = document.createElement("input");
      input.type = type;
      input.step = field === "speed" ? "0.01" : "0.1";
      input.value = getNested(human, field);
      input.addEventListener("input", () => {
        setNested(human, field, type === "number" ? Number(input.value) || 0 : input.value);
        if (field === "theta_deg") updateHumanGoalFromHeading(human);
        draw({ table: false });
      });
      cell.appendChild(input);
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function getNested(object, path) {
  return path.split(".").reduce((cursor, key) => cursor[key], object);
}

function setNested(object, path, value) {
  const parts = path.split(".");
  let cursor = object;
  for (let i = 0; i < parts.length - 1; i += 1) cursor = cursor[parts[i]];
  cursor[parts[parts.length - 1]] = value;
}

function buildScenario() {
  return {
    version: 1,
    name: state.name,
    time_step: state.time_step,
    max_time: state.max_time,
    human_motion_model: state.human_motion_model,
    human_dodge_robot: state.human_dodge_robot,
    robot: state.robot,
    humans: state.humans
  };
}

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e4e8ee";
  ctx.lineWidth = 1;
  for (let i = world.min; i <= world.max; i += 1) {
    const a = toScreen({ x: i, y: world.min });
    const b = toScreen({ x: i, y: world.max });
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();

    const c = toScreen({ x: world.min, y: i });
    const d = toScreen({ x: world.max, y: i });
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#9aa6b2";
  ctx.lineWidth = 1.5;
  const x0a = toScreen({ x: 0, y: world.min });
  const x0b = toScreen({ x: 0, y: world.max });
  ctx.beginPath();
  ctx.moveTo(x0a.x, x0a.y);
  ctx.lineTo(x0b.x, x0b.y);
  ctx.stroke();
  const y0a = toScreen({ x: world.min, y: 0 });
  const y0b = toScreen({ x: world.max, y: 0 });
  ctx.beginPath();
  ctx.moveTo(y0a.x, y0a.y);
  ctx.lineTo(y0b.x, y0b.y);
  ctx.stroke();
}

function drawCircle(point, radiusMeters, color, fill = true) {
  const p = toScreen(point);
  ctx.beginPath();
  ctx.arc(p.x, p.y, metersToPixels(radiusMeters), 0, Math.PI * 2);
  if (fill) {
    ctx.fillStyle = color;
    ctx.fill();
  } else {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function drawArrow(origin, thetaDeg, lengthMeters, color) {
  const radians = (thetaDeg * Math.PI) / 180;
  const a = toScreen(origin);
  const b = toScreen({
    x: origin.x + lengthMeters * Math.cos(radians),
    y: origin.y + lengthMeters * Math.sin(radians)
  });
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - 12 * Math.cos(angle - 0.45), b.y - 12 * Math.sin(angle - 0.45));
  ctx.lineTo(b.x - 12 * Math.cos(angle + 0.45), b.y - 12 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
}

function drawDashedLine(aPoint, bPoint, color) {
  const a = toScreen(aPoint);
  const b = toScreen(bPoint);
  ctx.save();
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function draw(options = {}) {
  readMainInputs();
  drawGrid();

  drawDashedLine(state.robot.position, state.robot.goal, "#d22f27");
  drawCircle(state.robot.goal, 0.12, "#d22f27", true);
  drawCircle(state.robot.position, robotRadius, "#e6c84f", true);
  drawArrow(state.robot.position, state.robot.theta_deg, 0.7, "#1d252d");

  state.humans.forEach((human, index) => {
    const color = index === selectedHuman ? "#b73a8c" : "#2457c5";
    const center = human.position;
    const p = toScreen(center);
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate((-human.theta_deg * Math.PI) / 180);
    ctx.fillStyle = "rgba(36, 87, 197, 0.10)";
    ctx.beginPath();
    ctx.ellipse(0, 0, metersToPixels(1.0), metersToPixels(0.75), 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    drawDashedLine(human.position, human.goal, color);
    drawCircle(human.goal, 0.09, color, true);
    drawCircle(human.position, humanRadius, color, true);
    drawArrow(human.position, human.theta_deg, 0.55, color);
    ctx.fillStyle = "#ffffff";
    ctx.font = "12px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(human.id, p.x, p.y + 4);
  });

  if (options.table !== false) renderHumanTable();
  updateTextOutputs();
  updateWarnings();
}

function updateTextOutputs() {
  const scenario = buildScenario();
  document.getElementById("jsonPreview").value = JSON.stringify(scenario, null, 2);
  const safeName = (state.name || "custom_scenario").replace(/[^a-zA-Z0-9_-]/g, "_");
  document.getElementById("evalCommand").value =
    `python evaluate_custom_scenario.py --scenario custom_scenarios/${safeName}.json ` +
    `--checkpoint checkpoints/sncp_ppo_v17.pt --output custom_eval/${safeName}.png ` +
    `--summary custom_eval/${safeName}.json`;
}

function updateWarnings() {
  const warnings = [];
  const robot = state.robot.position;
  state.humans.forEach((human, index) => {
    const d = Math.hypot(human.position.x - robot.x, human.position.y - robot.y);
    if (d < robotRadius + humanRadius) warnings.push(`${human.id}: starts in robot collision`);
    if (human.speed > robotMaxSpeed) warnings.push(`${human.id}: speed exceeds TurtleBot parity`);
    for (let j = index + 1; j < state.humans.length; j += 1) {
      const other = state.humans[j];
      const dh = Math.hypot(human.position.x - other.position.x, human.position.y - other.position.y);
      if (dh < humanRadius * 2) warnings.push(`${human.id}/${other.id}: overlapping humans`);
    }
  });
  document.getElementById("warnings").textContent = warnings.join(" | ");
}

function nearestTarget(point) {
  const candidates = [
    { type: "robot", point: state.robot.position },
    { type: "robotGoal", point: state.robot.goal }
  ];
  state.humans.forEach((human, index) => {
    candidates.push({ type: "human", index, point: human.position });
    candidates.push({ type: "humanGoal", index, point: human.goal });
  });
  let best = null;
  let bestDistance = Infinity;
  candidates.forEach((candidate) => {
    const screen = toScreen(candidate.point);
    const d = Math.hypot(screen.x - point.x, screen.y - point.y);
    if (d < bestDistance) {
      best = candidate;
      bestDistance = d;
    }
  });
  return bestDistance <= 28 ? best : null;
}

canvas.addEventListener("mousedown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  };
  dragTarget = nearestTarget(point);
  if (dragTarget && typeof dragTarget.index === "number") selectedHuman = dragTarget.index;
  draw();
});

window.addEventListener("mousemove", (event) => {
  if (!dragTarget) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const point = toWorld({
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  });
  point.x = round1(point.x);
  point.y = round1(point.y);

  if (dragTarget.type === "robot") state.robot.position = point;
  if (dragTarget.type === "robotGoal") state.robot.goal = point;
  if (dragTarget.type === "human") state.humans[dragTarget.index].position = point;
  if (dragTarget.type === "humanGoal") {
    const human = state.humans[dragTarget.index];
    human.goal = point;
    human.theta_deg = Math.round((Math.atan2(point.y - human.position.y, point.x - human.position.x) * 180) / Math.PI);
  }
  syncMainInputs();
  draw();
});

window.addEventListener("mouseup", () => {
  dragTarget = null;
});

Object.values(inputs).forEach((input) => {
  input.addEventListener("input", draw);
});

document.getElementById("addHuman").addEventListener("click", () => {
  const index = state.humans.length + 1;
  const human = {
    id: `h${index}`,
    position: { x: 0, y: 0 },
    theta_deg: 0,
    speed: 0.18,
    goal: { x: 6, y: 0 }
  };
  state.humans.push(human);
  selectedHuman = state.humans.length - 1;
  draw();
});

document.getElementById("removeHuman").addEventListener("click", () => {
  if (state.humans.length === 0) return;
  state.humans.splice(selectedHuman, 1);
  selectedHuman = clamp(selectedHuman, 0, Math.max(0, state.humans.length - 1));
  draw();
});

document.getElementById("exportJson").addEventListener("click", () => {
  const scenario = JSON.stringify(buildScenario(), null, 2);
  const blob = new Blob([scenario], { type: "application/json" });
  const link = document.createElement("a");
  const safeName = (state.name || "custom_scenario").replace(/[^a-zA-Z0-9_-]/g, "_");
  link.href = URL.createObjectURL(blob);
  link.download = `${safeName}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

document.getElementById("importJson").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const imported = JSON.parse(reader.result);
    state = {
      name: imported.name || "custom_scenario",
      time_step: imported.time_step || 0.25,
      max_time: imported.max_time || 50,
      human_motion_model: imported.human_motion_model || "linear",
      human_dodge_robot: Boolean(imported.human_dodge_robot),
      robot: imported.robot,
      humans: imported.humans || []
    };
    selectedHuman = 0;
    syncMainInputs();
    draw();
  };
  reader.readAsText(file);
});

document.getElementById("copyEvalCommand").addEventListener("click", async () => {
  const command = document.getElementById("evalCommand").value;
  await navigator.clipboard.writeText(command);
});

document.getElementById("loadPreset").addEventListener("click", () => {
  const preset = document.getElementById("presetSelect").value;
  if (preset === "crossing5") loadCrossing5();
  if (preset === "gate8") loadGate8();
  if (preset === "cluster10") loadCluster10();
  selectedHuman = 0;
  syncMainInputs();
  draw();
});

function loadCrossing5() {
  state.name = "custom_crossing5";
  state.human_motion_model = "linear";
  state.robot = { position: { x: -4, y: 0 }, goal: { x: 4, y: 0 }, theta_deg: 0 };
  state.humans = [
    { id: "h1", position: { x: 0, y: -2.4 }, theta_deg: 90, speed: 0.2, goal: { x: 0, y: 4 } },
    { id: "h2", position: { x: 0.7, y: 2.4 }, theta_deg: -90, speed: 0.2, goal: { x: 0.7, y: -4 } },
    { id: "h3", position: { x: -1.4, y: -1.5 }, theta_deg: 45, speed: 0.18, goal: { x: 3, y: 3 } },
    { id: "h4", position: { x: 1.5, y: 1.6 }, theta_deg: -135, speed: 0.18, goal: { x: -3, y: -3 } },
    { id: "h5", position: { x: -1.8, y: 1.4 }, theta_deg: -35, speed: 0.16, goal: { x: 3.3, y: -2.2 } }
  ];
}

function loadGate8() {
  state.name = "custom_gate8";
  state.human_motion_model = "linear";
  state.robot = { position: { x: -4.5, y: 0 }, goal: { x: 4.5, y: 0 }, theta_deg: 0 };
  state.humans = [];
  for (let i = 0; i < 8; i += 1) {
    const y = -2.8 + i * 0.8;
    const side = i % 2 === 0 ? -1 : 1;
    state.humans.push({
      id: `h${i + 1}`,
      position: { x: side * 0.9, y },
      theta_deg: side < 0 ? 0 : 180,
      speed: 0.12,
      goal: { x: -side * 0.9, y }
    });
  }
}

function loadCluster10() {
  state.name = "custom_cluster10";
  state.human_motion_model = "sfm";
  state.robot = { position: { x: -4, y: -1 }, goal: { x: 4, y: 1 }, theta_deg: 10 };
  state.humans = [];
  for (let i = 0; i < 10; i += 1) {
    const angle = (i / 10) * Math.PI * 2;
    const r = i % 2 === 0 ? 1.1 : 1.8;
    const x = round1(r * Math.cos(angle));
    const y = round1(r * Math.sin(angle));
    const theta = Math.round(((angle + Math.PI) * 180) / Math.PI);
    state.humans.push({
      id: `h${i + 1}`,
      position: { x, y },
      theta_deg: theta,
      speed: 0.18,
      goal: { x: round1(-x * 2), y: round1(-y * 2) }
    });
  }
}

syncMainInputs();
draw();
