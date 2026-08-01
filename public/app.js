const input = document.querySelector("#file-input");
const button = document.querySelector("#upload-button");
const dropZone = document.querySelector("#drop-zone");
const canvas = document.querySelector("#canvas");
const context = canvas.getContext("2d");
const empty = document.querySelector("#empty-state");
const loading = document.querySelector("#loading");
const status = document.querySelector("#status");
const list = document.querySelector("#detection-list");

button.addEventListener("click", () => input.click());
input.addEventListener("change", () => input.files[0] && detect(input.files[0]));
["dragenter", "dragover"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => dropZone.addEventListener(event, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => event.dataTransfer.files[0] && detect(event.dataTransfer.files[0]));

async function detect(file) {
  if (!file.type.startsWith("image/") || file.size > 10 * 1024 * 1024) { status.textContent = "Choose a supported image under 10 MB."; return; }
  loading.hidden = false;
  status.textContent = "Uploading image...";
  const image = await createImageBitmap(file);
  canvas.width = image.width; canvas.height = image.height;
  context.drawImage(image, 0, 0); canvas.hidden = false; empty.hidden = true;
  const form = new FormData(); form.append("file", file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    if (!response.ok) throw new Error((await response.json()).detail || "Inference failed");
    const result = await response.json();
    drawDetections(image, result.detections);
    document.querySelector("#object-count").textContent = result.detections.length;
    document.querySelector("#latency").textContent = `${result.inference_ms} ms`;
    document.querySelector("#resolution").textContent = `${result.image_width}x${result.image_height}`;
    list.innerHTML = result.detections.length ? result.detections.map((item) => `<li><span>${escapeHtml(item.class_name)}</span><strong>${(item.confidence * 100).toFixed(1)}%</strong></li>`).join("") : '<li class="placeholder">No objects found.</li>';
    status.textContent = "Inference complete";
  } catch (error) { status.textContent = error.message; }
  finally { loading.hidden = true; }
}

function drawDetections(image, detections) {
  context.drawImage(image, 0, 0); context.lineWidth = Math.max(2, image.width / 320); context.font = `${Math.max(14, image.width / 45)}px sans-serif`;
  detections.forEach((item) => { const [x1, y1, x2, y2] = item.box; const label = `${item.class_name} ${(item.confidence * 100).toFixed(0)}%`; context.strokeStyle = "#ff5638"; context.fillStyle = "#ff5638"; context.strokeRect(x1, y1, x2 - x1, y2 - y1); const width = context.measureText(label).width + 12; context.fillRect(x1, Math.max(0, y1 - 26), width, 26); context.fillStyle = "#fff"; context.fillText(label, x1 + 6, Math.max(18, y1 - 7)); });
}

function escapeHtml(value) { const span = document.createElement("span"); span.textContent = value; return span.innerHTML; }
