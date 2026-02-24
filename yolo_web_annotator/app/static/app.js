"use strict";

const el = {
  datasetDirInput: document.getElementById("datasetDirInput"),
  modelPathInput: document.getElementById("modelPathInput"),
  labelsDirInput: document.getElementById("labelsDirInput"),
  classesFileInput: document.getElementById("classesFileInput"),
  classesInput: document.getElementById("classesInput"),
  loadSessionBtn: document.getElementById("loadSessionBtn"),
  browseDatasetBtn: document.getElementById("browseDatasetBtn"),
  browseModelBtn: document.getElementById("browseModelBtn"),
  browseLabelsBtn: document.getElementById("browseLabelsBtn"),
  browseClassesBtn: document.getElementById("browseClassesBtn"),
  confInput: document.getElementById("confInput"),
  overlapCleanupCheck: document.getElementById("overlapCleanupCheck"),
  overlapThresholdInput: document.getElementById("overlapThresholdInput"),
  applyOverlapBtn: document.getElementById("applyOverlapBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  imageSelect: document.getElementById("imageSelect"),
  imageCounter: document.getElementById("imageCounter"),
  predictBtn: document.getElementById("predictBtn"),
  saveBtn: document.getElementById("saveBtn"),
  saveAllBtn: document.getElementById("saveAllBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  zoomFitBtn: document.getElementById("zoomFitBtn"),
  zoomLabel: document.getElementById("zoomLabel"),
  quickPredictBtn: document.getElementById("quickPredictBtn"),
  quickSaveBtn: document.getElementById("quickSaveBtn"),
  quickPrevBtn: document.getElementById("quickPrevBtn"),
  quickNextBtn: document.getElementById("quickNextBtn"),
  canvasWrap: document.getElementById("canvasWrap"),
  canvas: document.getElementById("annotCanvas"),
  statusBar: document.getElementById("statusBar"),
  classSelect: document.getElementById("classSelect"),
  deleteBtn: document.getElementById("deleteBtn"),
  clearBtn: document.getElementById("clearBtn"),
  boxList: document.getElementById("boxList"),
  pickerModal: document.getElementById("pickerModal"),
  pickerTitle: document.getElementById("pickerTitle"),
  pickerCloseBtn: document.getElementById("pickerCloseBtn"),
  pickerRootsBtn: document.getElementById("pickerRootsBtn"),
  pickerUpBtn: document.getElementById("pickerUpBtn"),
  pickerSelectCurrentBtn: document.getElementById("pickerSelectCurrentBtn"),
  pickerPath: document.getElementById("pickerPath"),
  pickerList: document.getElementById("pickerList"),
  classContextMenu: document.getElementById("classContextMenu"),
};

const ctx = el.canvas.getContext("2d");

const PICKER_CONFIG = {
  dataset: { title: "Select Dataset Folder", mode: "dir", input: el.datasetDirInput, allowFolder: true },
  model: { title: "Select Model File", mode: "model", input: el.modelPathInput, allowFile: true },
  labels: { title: "Select Labels Folder", mode: "dir", input: el.labelsDirInput, allowFolder: true },
  classes: { title: "Select Classes YAML/TXT", mode: "yaml", input: el.classesFileInput, allowFile: true },
};

const state = {
  sessionLoaded: false,
  images: [],
  imageIndex: -1,
  imagePath: "",
  image: null,
  classes: [],
  defaultClassId: 0,
  boxes: [],
  annotationsByPath: {},
  dirtyPaths: new Set(),
  selectedIndex: -1,
  interaction: null,
  dirty: false,
  historyByPath: {},
  contextMenuBoxIndex: -1,
  classMenuQuery: "",
  clipboardBoxes: [],
  clipboardPasteCount: 0,
  loadToken: 0,
  view: { width: 0, height: 0, fitScale: 1, scale: 1, offsetX: 0, offsetY: 0, userZoom: 1, lastPointer: null },
  picker: { target: null, currentPath: null, parentPath: null },
};

function setStatus(message, isError = false) {
  el.statusBar.textContent = message;
  el.statusBar.classList.toggle("error", isError);
}

function parseClassesFromInput() {
  return el.classesInput.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function deepCopyBoxes(boxes) {
  return boxes.map((box) => ({
    class_id: box.class_id,
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    score: typeof box.score === "number" ? box.score : undefined,
  }));
}

function getHistoryForPath(path) {
  if (!state.historyByPath[path]) {
    state.historyByPath[path] = [];
  }
  return state.historyByPath[path];
}

function pushHistoryForCurrent() {
  if (!state.imagePath) {
    return;
  }
  const history = getHistoryForPath(state.imagePath);
  history.push({
    boxes: deepCopyBoxes(state.boxes),
    selectedIndex: state.selectedIndex,
  });
  if (history.length > 80) {
    history.shift();
  }
}

function undoCurrent() {
  if (!state.imagePath) {
    return false;
  }
  const history = getHistoryForPath(state.imagePath);
  if (history.length === 0) {
    return false;
  }
  const snapshot = history.pop();
  state.boxes = deepCopyBoxes(snapshot.boxes);
  state.selectedIndex = snapshot.selectedIndex;
  refreshBoxList();
  render();
  updateCacheForCurrent(true);
  return true;
}

function overlapRatio(boxA, boxB) {
  const left = Math.max(boxA.x, boxB.x);
  const top = Math.max(boxA.y, boxB.y);
  const right = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const bottom = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);
  const interW = Math.max(0, right - left);
  const interH = Math.max(0, bottom - top);
  const interArea = interW * interH;
  if (interArea <= 0) {
    return 0;
  }
  const areaA = boxA.width * boxA.height;
  const areaB = boxB.width * boxB.height;
  const union = areaA + areaB - interArea;
  if (union <= 0) {
    return 0;
  }
  return interArea / union;
}

function chooseBoxToDrop(boxA, boxB, indexA, indexB) {
  const hasA = typeof boxA.score === "number";
  const hasB = typeof boxB.score === "number";
  if (hasA && hasB) {
    return boxA.score <= boxB.score ? indexA : indexB;
  }
  if (hasA && !hasB) {
    return indexB;
  }
  if (!hasA && hasB) {
    return indexA;
  }
  return indexB;
}

function getOverlapThresholdNormalized() {
  if (!el.overlapThresholdInput) {
    return 0.5;
  }
  const raw = Number(el.overlapThresholdInput.value);
  const safe = Number.isFinite(raw) ? clamp(raw, 1, 100) : 50;
  el.overlapThresholdInput.value = String(Math.round(safe));
  return safe / 100;
}

function isOverlapCleanupEnabled() {
  return Boolean(el.overlapCleanupCheck && el.overlapCleanupCheck.checked);
}

function removeOverlaps(boxes, threshold) {
  if (!Array.isArray(boxes) || boxes.length < 2) {
    return deepCopyBoxes(boxes || []);
  }
  const kept = deepCopyBoxes(boxes);
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < kept.length; i += 1) {
      for (let j = i + 1; j < kept.length; j += 1) {
        if (overlapRatio(kept[i], kept[j]) >= threshold) {
          const dropIndex = chooseBoxToDrop(kept[i], kept[j], i, j);
          kept.splice(dropIndex, 1);
          changed = true;
          break;
        }
      }
      if (changed) {
        break;
      }
    }
  }
  return kept;
}

function cleanupCurrentOverlaps() {
  if (!state.imagePath || state.boxes.length < 2) {
    return 0;
  }
  const threshold = getOverlapThresholdNormalized();
  const cleaned = removeOverlaps(state.boxes, threshold);
  const removed = state.boxes.length - cleaned.length;
  if (removed > 0) {
    pushHistoryForCurrent();
    state.boxes = cleaned;
    state.selectedIndex = Math.min(state.selectedIndex, state.boxes.length - 1);
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
  }
  return removed;
}

function maybeCleanupBoxes(boxes) {
  if (!isOverlapCleanupEnabled()) {
    return deepCopyBoxes(boxes);
  }
  return removeOverlaps(boxes, getOverlapThresholdNormalized());
}

function hideClassContextMenu() {
  state.contextMenuBoxIndex = -1;
  state.classMenuQuery = "";
  if (!el.classContextMenu) {
    return;
  }
  el.classContextMenu.classList.add("hidden");
  el.classContextMenu.innerHTML = "";
}

function applyClassFromContextMenu(classId) {
  if (!Number.isFinite(classId)) {
    return;
  }
  if (state.contextMenuBoxIndex >= 0 && state.contextMenuBoxIndex < state.boxes.length) {
    pushHistoryForCurrent();
    state.boxes[state.contextMenuBoxIndex].class_id = classId;
    state.selectedIndex = state.contextMenuBoxIndex;
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
  }
  state.defaultClassId = classId;
  if (el.classSelect) {
    el.classSelect.value = String(classId);
  }
}

function getFilteredClassItems(query) {
  const needle = String(query || "").trim().toLowerCase();
  const items = state.classes.map((name, classId) => ({ classId, name: String(name) }));
  if (!needle) {
    return items;
  }
  return items.filter((item) => item.name.toLowerCase().includes(needle) || String(item.classId).startsWith(needle));
}

function renderClassMenuList(query, listContainer) {
  if (!listContainer) {
    return;
  }
  listContainer.innerHTML = "";
  const items = getFilteredClassItems(query);
  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "menu-item menu-empty";
    empty.textContent = "No classes";
    listContainer.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "menu-item";
    button.dataset.classId = String(item.classId);
    button.title = `${item.classId}: ${item.name}`;
    button.textContent = `${item.classId}: ${item.name}`;
    button.addEventListener("click", () => {
      applyClassFromContextMenu(item.classId);
      hideClassContextMenu();
    });
    listContainer.appendChild(button);
  });
}

function getClassMenuSearchInput() {
  if (!el.classContextMenu) {
    return null;
  }
  return el.classContextMenu.querySelector(".class-menu-search");
}

function updateClassMenuSearchValue(nextValue) {
  const input = getClassMenuSearchInput();
  if (!input) {
    return false;
  }
  input.focus();
  input.value = nextValue;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  return true;
}

function chooseFirstVisibleClassFromMenu() {
  if (!el.classContextMenu) {
    return false;
  }
  const first = el.classContextMenu.querySelector("button.menu-item[data-class-id]");
  if (!first) {
    return false;
  }
  const classId = Number(first.dataset.classId);
  if (!Number.isFinite(classId)) {
    return false;
  }
  applyClassFromContextMenu(classId);
  hideClassContextMenu();
  return true;
}

function showClassContextMenu(clientX, clientY, boxIndex, initialQuery = "") {
  if (!el.classContextMenu) {
    return;
  }
  state.contextMenuBoxIndex = boxIndex;
  state.classMenuQuery = String(initialQuery || "");
  el.classContextMenu.innerHTML = "";

  const inner = document.createElement("div");
  inner.className = "class-menu-inner";

  const searchInput = document.createElement("input");
  searchInput.type = "text";
  searchInput.className = "class-menu-search";
  searchInput.placeholder = "Search by id or name";
  searchInput.value = state.classMenuQuery;

  const listContainer = document.createElement("div");
  listContainer.className = "class-menu-list";

  const rerender = () => {
    state.classMenuQuery = searchInput.value;
    renderClassMenuList(state.classMenuQuery, listContainer);
  };

  searchInput.addEventListener("input", rerender);
  searchInput.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      hideClassContextMenu();
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      chooseFirstVisibleClassFromMenu();
    }
  });

  inner.appendChild(searchInput);
  inner.appendChild(listContainer);
  el.classContextMenu.appendChild(inner);
  rerender();

  el.classContextMenu.classList.remove("hidden");
  const menuWidth = el.classContextMenu.offsetWidth || 190;
  const menuHeight = el.classContextMenu.offsetHeight || 220;
  const left = clamp(clientX, 4, window.innerWidth - menuWidth - 4);
  const top = clamp(clientY, 4, window.innerHeight - menuHeight - 4);
  el.classContextMenu.style.left = `${left}px`;
  el.classContextMenu.style.top = `${top}px`;

  requestAnimationFrame(() => {
    if (!el.classContextMenu || el.classContextMenu.classList.contains("hidden")) {
      return;
    }
    searchInput.focus();
    if (state.classMenuQuery) {
      searchInput.setSelectionRange(searchInput.value.length, searchInput.value.length);
    } else {
      searchInput.select();
    }
  });
}

function classNameById(classId) {
  return state.classes[classId] || `class_${classId}`;
}

function ensureClassCapacity(maxClassId) {
  if (maxClassId < 0) {
    return;
  }
  while (state.classes.length <= maxClassId) {
    state.classes.push(`class_${state.classes.length}`);
  }
  if (state.classes.length === 0) {
    state.classes.push("class_0");
  }
}

function syncClassControls() {
  if (state.classes.length === 0) {
    state.classes = ["class_0"];
  }
  const prev = Number(el.classSelect.value);
  el.classSelect.innerHTML = "";
  state.classes.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index}: ${name}`;
    el.classSelect.appendChild(option);
  });
  if (Number.isFinite(prev) && prev >= 0 && prev < state.classes.length) {
    state.defaultClassId = prev;
  } else if (state.defaultClassId >= state.classes.length) {
    state.defaultClassId = 0;
  }
  el.classSelect.value = String(state.defaultClassId);
}

function syncImageSelect() {
  el.imageSelect.innerHTML = "";
  state.images.forEach((path, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = path;
    el.imageSelect.appendChild(option);
  });
  if (state.imageIndex >= 0 && state.imageIndex < state.images.length) {
    el.imageSelect.value = String(state.imageIndex);
  }
  updateImageCounter();
}

function updateImageCounter() {
  if (state.images.length === 0 || state.imageIndex < 0) {
    el.imageCounter.textContent = "0 / 0";
    return;
  }
  el.imageCounter.textContent = `${state.imageIndex + 1} / ${state.images.length}`;
}

function updateZoomLabel() {
  el.zoomLabel.textContent = `${Math.round(state.view.userZoom * 100)}%`;
}

function updateStatusForCurrent(extra = "") {
  if (!state.imagePath) {
    return;
  }
  const dirty = state.dirty ? " (unsaved)" : "";
  const tail = extra ? ` | ${extra}` : "";
  setStatus(
    `${state.imagePath} | ${state.imageIndex + 1}/${state.images.length} | boxes: ${state.boxes.length}${dirty} | unsaved: ${state.dirtyPaths.size}${tail}`
  );
}

function setCurrentDirty(value) {
  if (!state.imagePath) {
    return;
  }
  if (value) {
    state.dirtyPaths.add(state.imagePath);
  } else {
    state.dirtyPaths.delete(state.imagePath);
  }
  state.dirty = state.dirtyPaths.has(state.imagePath);
  updateStatusForCurrent();
}

function updateCacheForCurrent(isDirty = true) {
  if (!state.imagePath) {
    return;
  }
  state.annotationsByPath[state.imagePath] = deepCopyBoxes(state.boxes);
  setCurrentDirty(isDirty);
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function colorForClass(classId) {
  const hue = (classId * 57 + 29) % 360;
  return `hsl(${hue}, 78%, 46%)`;
}

function refreshCanvasCursor() {
  if (!el.canvas) {
    return;
  }
  if (state.interaction && state.interaction.type === "pan") {
    el.canvas.style.cursor = "grabbing";
    return;
  }
  el.canvas.style.cursor = "default";
}

function resizeCanvas() {
  const rect = el.canvasWrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));

  el.canvas.width = Math.floor(width * dpr);
  el.canvas.height = Math.floor(height * dpr);
  el.canvas.style.width = `${width}px`;
  el.canvas.style.height = `${height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.view.width = width;
  state.view.height = height;
  if (state.image) {
    fitImageInCanvas(state.view.userZoom);
  }
  render();
}

function clampViewOffset() {
  if (!state.image) {
    return;
  }
  const drawW = state.image.naturalWidth * state.view.scale;
  const drawH = state.image.naturalHeight * state.view.scale;
  if (drawW <= state.view.width) {
    state.view.offsetX = (state.view.width - drawW) / 2;
  } else {
    state.view.offsetX = clamp(state.view.offsetX, state.view.width - drawW, 0);
  }
  if (drawH <= state.view.height) {
    state.view.offsetY = (state.view.height - drawH) / 2;
  } else {
    state.view.offsetY = clamp(state.view.offsetY, state.view.height - drawH, 0);
  }
}

function applyScaleAroundPoint(userZoom, anchorCanvasPoint) {
  if (!state.image) {
    return;
  }
  const oldScale = state.view.scale || state.view.fitScale * state.view.userZoom || 1;
  const anchor = anchorCanvasPoint || { x: state.view.width / 2, y: state.view.height / 2 };
  const imagePoint = {
    x: (anchor.x - state.view.offsetX) / oldScale,
    y: (anchor.y - state.view.offsetY) / oldScale,
  };

  state.view.userZoom = clamp(userZoom, 0.1, 8);
  state.view.scale = state.view.fitScale * state.view.userZoom;
  state.view.offsetX = anchor.x - imagePoint.x * state.view.scale;
  state.view.offsetY = anchor.y - imagePoint.y * state.view.scale;
  clampViewOffset();
  updateZoomLabel();
}

function fitImageInCanvas(keepZoom = 1) {
  if (!state.image) {
    return;
  }
  const imageWidth = state.image.naturalWidth;
  const imageHeight = state.image.naturalHeight;
  state.view.fitScale = Math.min(state.view.width / imageWidth, state.view.height / imageHeight);
  state.view.userZoom = clamp(keepZoom, 0.1, 8);
  state.view.scale = state.view.fitScale * state.view.userZoom;
  state.view.offsetX = (state.view.width - imageWidth * state.view.scale) / 2;
  state.view.offsetY = (state.view.height - imageHeight * state.view.scale) / 2;
  clampViewOffset();
  updateZoomLabel();
}

function setZoom(zoom, anchorCanvasPoint = null) {
  if (!state.image) {
    state.view.userZoom = clamp(zoom, 0.1, 8);
    updateZoomLabel();
    return;
  }
  const anchor = anchorCanvasPoint || state.view.lastPointer || { x: state.view.width / 2, y: state.view.height / 2 };
  applyScaleAroundPoint(zoom, anchor);
  render();
}

function zoomBy(factor, anchorCanvasPoint = null) {
  setZoom(state.view.userZoom * factor, anchorCanvasPoint);
}

function zoomFit() {
  if (!state.image) {
    state.view.userZoom = 1;
    updateZoomLabel();
    return;
  }
  fitImageInCanvas(1);
  render();
}

function imageToCanvas(point) {
  return {
    x: point.x * state.view.scale + state.view.offsetX,
    y: point.y * state.view.scale + state.view.offsetY,
  };
}

function canvasToImage(point) {
  return {
    x: (point.x - state.view.offsetX) / state.view.scale,
    y: (point.y - state.view.offsetY) / state.view.scale,
  };
}

function getPointer(event) {
  const rect = el.canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function clampPointToImage(point) {
  if (!state.image) {
    return point;
  }
  const maxX = state.image.naturalWidth;
  const maxY = state.image.naturalHeight;
  return { x: clamp(point.x, 0, maxX), y: clamp(point.y, 0, maxY) };
}

function clampBox(box) {
  if (!state.image) {
    return box;
  }
  const imgW = state.image.naturalWidth;
  const imgH = state.image.naturalHeight;
  const safe = { ...box };
  safe.width = clamp(safe.width, 1, imgW);
  safe.height = clamp(safe.height, 1, imgH);
  safe.x = clamp(safe.x, 0, imgW - safe.width);
  safe.y = clamp(safe.y, 0, imgH - safe.height);
  return safe;
}

function normalizedRect(a, b, classId) {
  const left = Math.min(a.x, b.x);
  const top = Math.min(a.y, b.y);
  const right = Math.max(a.x, b.x);
  const bottom = Math.max(a.y, b.y);
  return { class_id: classId, x: left, y: top, width: right - left, height: bottom - top };
}

function hitBox(point) {
  for (let i = state.boxes.length - 1; i >= 0; i -= 1) {
    const box = state.boxes[i];
    if (
      point.x >= box.x &&
      point.x <= box.x + box.width &&
      point.y >= box.y &&
      point.y <= box.y + box.height
    ) {
      return i;
    }
  }
  return -1;
}

function hitHandle(box, point) {
  const radius = 8 / state.view.scale;
  const corners = {
    nw: { x: box.x, y: box.y },
    ne: { x: box.x + box.width, y: box.y },
    sw: { x: box.x, y: box.y + box.height },
    se: { x: box.x + box.width, y: box.y + box.height },
  };
  for (const [name, corner] of Object.entries(corners)) {
    const dx = point.x - corner.x;
    const dy = point.y - corner.y;
    if (Math.sqrt(dx * dx + dy * dy) <= radius) {
      return name;
    }
  }
  return null;
}

function applyResize(startBox, handle, point) {
  let left = startBox.x;
  let top = startBox.y;
  let right = startBox.x + startBox.width;
  let bottom = startBox.y + startBox.height;
  if (handle.includes("n")) {
    top = point.y;
  }
  if (handle.includes("s")) {
    bottom = point.y;
  }
  if (handle.includes("w")) {
    left = point.x;
  }
  if (handle.includes("e")) {
    right = point.x;
  }
  return clampBox(normalizedRect({ x: left, y: top }, { x: right, y: bottom }, startBox.class_id));
}

function drawHandles(box, color) {
  const points = [
    { x: box.x, y: box.y },
    { x: box.x + box.width, y: box.y },
    { x: box.x, y: box.y + box.height },
    { x: box.x + box.width, y: box.y + box.height },
  ];
  ctx.fillStyle = color;
  points.forEach((point) => {
    const canvasPoint = imageToCanvas(point);
    ctx.fillRect(canvasPoint.x - 4.5, canvasPoint.y - 4.5, 9, 9);
  });
}

function drawBox(box, selected = false, isDraft = false) {
  const topLeft = imageToCanvas({ x: box.x, y: box.y });
  const width = box.width * state.view.scale;
  const height = box.height * state.view.scale;
  const color = colorForClass(box.class_id);

  ctx.save();
  ctx.lineWidth = selected ? 2.4 : 1.7;
  if (isDraft) {
    ctx.setLineDash([8, 6]);
  }
  ctx.strokeStyle = color;
  ctx.fillStyle = selected ? "rgba(6, 115, 106, 0.14)" : "rgba(6, 115, 106, 0.08)";
  ctx.fillRect(topLeft.x, topLeft.y, width, height);
  ctx.strokeRect(topLeft.x, topLeft.y, width, height);
  ctx.setLineDash([]);

  const label = `${box.class_id}: ${classNameById(box.class_id)}`;
  ctx.font = '12px "Trebuchet MS", "Lucida Sans Unicode", sans-serif';
  const textWidth = ctx.measureText(label).width;
  const textX = topLeft.x + 2;
  const textY = Math.max(14, topLeft.y - 4);
  ctx.fillStyle = color;
  ctx.fillRect(textX - 2, textY - 12, textWidth + 7, 14);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, textX + 1, textY - 1);
  if (selected) {
    drawHandles(box, color);
  }
  ctx.restore();
}

function render() {
  refreshCanvasCursor();
  ctx.clearRect(0, 0, state.view.width, state.view.height);
  if (!state.image) {
    return;
  }
  const drawWidth = state.image.naturalWidth * state.view.scale;
  const drawHeight = state.image.naturalHeight * state.view.scale;
  ctx.drawImage(state.image, state.view.offsetX, state.view.offsetY, drawWidth, drawHeight);
  state.boxes.forEach((box, index) => drawBox(box, index === state.selectedIndex));
  if (state.interaction && state.interaction.type === "draw") {
    drawBox(normalizedRect(state.interaction.start, state.interaction.current, state.defaultClassId), false, true);
  }
}

function refreshBoxList() {
  el.boxList.innerHTML = "";
  state.boxes.forEach((box, index) => {
    const li = document.createElement("li");
    if (index === state.selectedIndex) {
      li.classList.add("active");
    }
    li.dataset.index = String(index);
    const scorePart = typeof box.score === "number" ? ` | score ${box.score.toFixed(2)}` : "";
    li.textContent =
      `#${index + 1} ${classNameById(box.class_id)} ` +
      `[${round1(box.x)}, ${round1(box.y)}, ${round1(box.width)}, ${round1(box.height)}]${scorePart}`;
    el.boxList.appendChild(li);
  });
}

function selectBox(index) {
  if (index < 0 || index >= state.boxes.length) {
    state.selectedIndex = -1;
    refreshBoxList();
    render();
    return;
  }
  state.selectedIndex = index;
  ensureClassCapacity(state.boxes[index].class_id);
  el.classSelect.value = String(state.boxes[index].class_id);
  refreshBoxList();
  render();
}

function deleteSelectedBox() {
  if (state.selectedIndex < 0) {
    return;
  }
  pushHistoryForCurrent();
  state.boxes.splice(state.selectedIndex, 1);
  state.selectedIndex = Math.min(state.selectedIndex, state.boxes.length - 1);
  updateCacheForCurrent(true);
  refreshBoxList();
  render();
}

function moveSelectedBox(dx, dy) {
  if (state.selectedIndex < 0) {
    return;
  }
  pushHistoryForCurrent();
  const box = state.boxes[state.selectedIndex];
  state.boxes[state.selectedIndex] = clampBox({ ...box, x: box.x + dx, y: box.y + dy });
  updateCacheForCurrent(true);
  refreshBoxList();
  render();
}

function copySelectedBoxToClipboard() {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.boxes.length) {
    return false;
  }
  state.clipboardBoxes = deepCopyBoxes([state.boxes[state.selectedIndex]]);
  state.clipboardPasteCount = 0;
  return true;
}

function pasteBoxesFromClipboard() {
  if (!state.image || !Array.isArray(state.clipboardBoxes) || state.clipboardBoxes.length === 0) {
    return 0;
  }

  state.clipboardPasteCount += 1;
  const offsetStep = 12 * state.clipboardPasteCount;
  const pasted = [];
  state.clipboardBoxes.forEach((box) => {
    const classId = Number.isFinite(box.class_id) ? Math.max(0, Math.floor(box.class_id)) : 0;
    ensureClassCapacity(classId);
    const pastedBox = clampBox({
      class_id: classId,
      x: Number(box.x) + offsetStep,
      y: Number(box.y) + offsetStep,
      width: Number(box.width),
      height: Number(box.height),
      score: undefined,
    });
    pasted.push(pastedBox);
  });

  if (pasted.length === 0) {
    return 0;
  }

  pushHistoryForCurrent();
  state.boxes.push(...pasted);
  state.selectedIndex = state.boxes.length - 1;
  updateCacheForCurrent(true);
  refreshBoxList();
  render();
  return pasted.length;
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function sanitizeBoxes(rawBoxes) {
  const result = [];
  let maxClassId = 0;
  rawBoxes.forEach((raw) => {
    const classId = Number.isFinite(raw.class_id) ? Math.max(0, Math.floor(raw.class_id)) : 0;
    const box = clampBox({
      class_id: classId,
      x: Math.max(0, Number(raw.x) || 0),
      y: Math.max(0, Number(raw.y) || 0),
      width: Math.max(1, Number(raw.width) || 1),
      height: Math.max(1, Number(raw.height) || 1),
      score: typeof raw.score === "number" ? raw.score : undefined,
    });
    result.push(box);
    maxClassId = Math.max(maxClassId, classId);
  });
  ensureClassCapacity(maxClassId);
  return result;
}

function loadImageElement(path) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Image loading failed."));
    image.src = `/api/image?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  });
}

async function loadSession() {
  const datasetDir = el.datasetDirInput.value.trim();
  if (!datasetDir) {
    setStatus("Dataset directory is required.", true);
    return;
  }

  const payload = {
    dataset_dir: datasetDir,
    classes: parseClassesFromInput(),
    classes_file: el.classesFileInput.value.trim() || null,
    model_path: el.modelPathInput.value.trim() || null,
    labels_dir: el.labelsDirInput.value.trim() || null,
  };

  setStatus("Loading dataset...");
  try {
    const data = await apiJson("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    state.sessionLoaded = true;
    state.images = data.images || [];
    state.classes = data.classes || [];
    state.annotationsByPath = {};
    state.dirtyPaths = new Set();
    state.historyByPath = {};
    state.selectedIndex = -1;
    state.interaction = null;
    hideClassContextMenu();
    if (state.classes.length > 0) {
      el.classesInput.value = state.classes.join("\n");
    }
    if (data.model_path) {
      el.modelPathInput.value = data.model_path;
    }
    if (data.labels_dir) {
      el.labelsDirInput.value = data.labels_dir;
    }

    syncClassControls();
    syncImageSelect();

    if (state.images.length === 0) {
      setStatus("No images found.", true);
      return;
    }

    await loadImageByIndex(0);
    setStatus(`Dataset loaded: ${state.images.length} images.`);
  } catch (error) {
    setStatus(`Load failed: ${error.message}`, true);
  }
}

async function loadImageByIndex(index) {
  if (!state.sessionLoaded) {
    return;
  }
  if (index < 0 || index >= state.images.length) {
    return;
  }

  const token = ++state.loadToken;
  const imagePath = state.images[index];
  state.imageIndex = index;
  state.imagePath = imagePath;
  if (!state.historyByPath[imagePath]) {
    state.historyByPath[imagePath] = [];
  }
  hideClassContextMenu();
  syncImageSelect();
  setStatus(`Loading ${imagePath}...`);

  try {
    const imagePromise = loadImageElement(imagePath);
    let boxesPromise = null;
    if (!state.annotationsByPath[imagePath]) {
      boxesPromise = apiJson(`/api/annotations?path=${encodeURIComponent(imagePath)}`);
    }
    const image = await imagePromise;
    if (token !== state.loadToken) {
      return;
    }
    if (boxesPromise) {
      const ann = await boxesPromise;
      if (token !== state.loadToken) {
        return;
      }
      state.annotationsByPath[imagePath] = sanitizeBoxes(ann.boxes || []);
      if (!state.historyByPath[imagePath]) {
        state.historyByPath[imagePath] = [];
      }
    }

    state.image = image;
    state.boxes = deepCopyBoxes(state.annotationsByPath[imagePath] || []);
    state.selectedIndex = -1;
    state.interaction = null;
    state.dirty = state.dirtyPaths.has(imagePath);
    fitImageInCanvas();
    syncClassControls();
    refreshBoxList();
    render();
    updateStatusForCurrent();
  } catch (error) {
    setStatus(`Image load failed: ${error.message}`, true);
  }
}

async function saveCurrentImage() {
  if (!state.imagePath) {
    return;
  }
  let boxesToSave = deepCopyBoxes(state.boxes);
  if (isOverlapCleanupEnabled()) {
    boxesToSave = maybeCleanupBoxes(boxesToSave);
    if (boxesToSave.length !== state.boxes.length) {
      pushHistoryForCurrent();
      state.boxes = deepCopyBoxes(boxesToSave);
      state.selectedIndex = Math.min(state.selectedIndex, state.boxes.length - 1);
      updateCacheForCurrent(true);
      refreshBoxList();
      render();
    }
  }
  try {
    await apiJson(`/api/annotations?path=${encodeURIComponent(state.imagePath)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: boxesToSave }),
    });
    state.annotationsByPath[state.imagePath] = deepCopyBoxes(boxesToSave);
    setCurrentDirty(false);
    setStatus(`Saved: ${state.imagePath}`);
  } catch (error) {
    setStatus(`Save failed: ${error.message}`, true);
  }
}

async function saveAllDirty() {
  const dirtyPaths = Array.from(state.dirtyPaths);
  if (dirtyPaths.length === 0) {
    setStatus("No unsaved changes.");
    return;
  }

  const items = dirtyPaths.map((path) => {
    const rawBoxes = state.annotationsByPath[path] || [];
    const boxes = isOverlapCleanupEnabled() ? maybeCleanupBoxes(rawBoxes) : deepCopyBoxes(rawBoxes);
    state.annotationsByPath[path] = deepCopyBoxes(boxes);
    if (path === state.imagePath) {
      state.boxes = deepCopyBoxes(boxes);
      state.selectedIndex = Math.min(state.selectedIndex, state.boxes.length - 1);
    }
    return { path, boxes };
  });
  if (state.imagePath) {
    refreshBoxList();
    render();
  }
  setStatus(`Saving ${items.length} changed images...`);
  try {
    const result = await apiJson("/api/annotations/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });

    const failed = new Set((result.errors || []).map((item) => item.path));
    dirtyPaths.forEach((path) => {
      if (!failed.has(path)) {
        state.dirtyPaths.delete(path);
      }
    });
    state.dirty = state.dirtyPaths.has(state.imagePath);
    updateStatusForCurrent(`save_all: ${result.saved_count} saved, ${result.error_count} failed`);
  } catch (error) {
    setStatus(`Save All failed: ${error.message}`, true);
  }
}

async function predictCurrentImage() {
  if (!state.imagePath) {
    setStatus("No image selected.", true);
    return;
  }
  const payload = {
    model_path: el.modelPathInput.value.trim() || null,
    conf: Number(el.confInput.value) || 0.25,
  };

  setStatus("Running YOLO prediction...");
  try {
    const data = await apiJson(`/api/predict?path=${encodeURIComponent(state.imagePath)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (Array.isArray(data.classes) && data.classes.length > 0) {
      state.classes = data.classes;
      el.classesInput.value = state.classes.join("\n");
      syncClassControls();
    }
    pushHistoryForCurrent();
    const predicted = sanitizeBoxes(data.boxes || []);
    state.boxes = maybeCleanupBoxes(predicted);
    state.selectedIndex = -1;
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
    setStatus(`Prediction complete: ${state.boxes.length} boxes.`);
  } catch (error) {
    setStatus(`Prediction failed: ${error.message}`, true);
  }
}

function onCanvasMouseDown(event) {
  if (!state.image) {
    return;
  }
  if (el.canvas && typeof el.canvas.focus === "function") {
    el.canvas.focus();
  }
  hideClassContextMenu();
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;

  if (event.button === 1) {
    event.preventDefault();
    state.interaction = {
      type: "pan",
      startCanvas: pointerCanvas,
      originOffsetX: state.view.offsetX,
      originOffsetY: state.view.offsetY,
    };
    render();
    return;
  }
  if (event.button !== 0) {
    return;
  }

  const point = clampPointToImage(canvasToImage(pointerCanvas));

  if (state.selectedIndex >= 0) {
    const selectedBox = state.boxes[state.selectedIndex];
    const handle = hitHandle(selectedBox, point);
    if (handle) {
      pushHistoryForCurrent();
      state.interaction = {
        type: "resize",
        boxIndex: state.selectedIndex,
        handle,
        startBox: { ...selectedBox },
      };
      return;
    }
  }

  const boxIndex = hitBox(point);
  if (boxIndex >= 0) {
    selectBox(boxIndex);
    const originBox = state.boxes[boxIndex];
    pushHistoryForCurrent();
    state.interaction = { type: "move", boxIndex, startPoint: point, originBox: { ...originBox } };
    return;
  }

  state.selectedIndex = -1;
  refreshBoxList();
  state.interaction = { type: "draw", start: point, current: point };
  render();
}

function onCanvasMouseMove(event) {
  if (!state.image) {
    return;
  }
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;
  if (!state.interaction) {
    return;
  }

  if (state.interaction.type === "pan") {
    const pan = state.interaction;
    const dx = pointerCanvas.x - pan.startCanvas.x;
    const dy = pointerCanvas.y - pan.startCanvas.y;
    state.view.offsetX = pan.originOffsetX + dx;
    state.view.offsetY = pan.originOffsetY + dy;
    clampViewOffset();
    render();
    return;
  }

  const pointer = clampPointToImage(canvasToImage(pointerCanvas));

  if (state.interaction.type === "draw") {
    state.interaction.current = pointer;
    render();
    return;
  }

  if (state.interaction.type === "move") {
    const move = state.interaction;
    const dx = pointer.x - move.startPoint.x;
    const dy = pointer.y - move.startPoint.y;
    state.boxes[move.boxIndex] = clampBox({ ...move.originBox, x: move.originBox.x + dx, y: move.originBox.y + dy });
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
    return;
  }

  if (state.interaction.type === "resize") {
    const resize = state.interaction;
    state.boxes[resize.boxIndex] = applyResize(resize.startBox, resize.handle, pointer);
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
  }
}

function onCanvasMouseUp() {
  if (!state.interaction) {
    return;
  }
  if (state.interaction.type === "draw") {
    const draft = normalizedRect(state.interaction.start, state.interaction.current, state.defaultClassId);
    if (draft.width >= 3 && draft.height >= 3) {
      pushHistoryForCurrent();
      ensureClassCapacity(draft.class_id);
      state.boxes.push(clampBox(draft));
      state.selectedIndex = state.boxes.length - 1;
      updateCacheForCurrent(true);
    }
  }
  state.interaction = null;
  refreshBoxList();
  render();
}

function onCanvasWheel(event) {
  if (!state.image) {
    return;
  }
  event.preventDefault();
  const pointer = getPointer(event);
  state.view.lastPointer = pointer;
  zoomBy(event.deltaY < 0 ? 1.1 : 0.9, pointer);
}

function onCanvasContextMenu(event) {
  if (!state.image) {
    return;
  }
  if (el.canvas && typeof el.canvas.focus === "function") {
    el.canvas.focus();
  }
  event.preventDefault();
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;
  const point = clampPointToImage(canvasToImage(pointerCanvas));
  const boxIndex = hitBox(point);
  if (boxIndex >= 0) {
    selectBox(boxIndex);
    showClassContextMenu(event.clientX, event.clientY, boxIndex);
  } else {
    state.selectedIndex = -1;
    refreshBoxList();
    showClassContextMenu(event.clientX, event.clientY, -1);
  }
}

function onCanvasDoubleClick(event) {
  if (!state.image) {
    return;
  }
  if (event.button !== 0) {
    return;
  }
  const pointerCanvas = getPointer(event);
  state.view.lastPointer = pointerCanvas;
  const point = clampPointToImage(canvasToImage(pointerCanvas));
  const boxIndex = hitBox(point);
  if (boxIndex < 0) {
    return;
  }
  selectBox(boxIndex);
  showClassContextMenu(event.clientX, event.clientY, boxIndex);
}

function onBoxListClick(event) {
  hideClassContextMenu();
  const item = event.target.closest("li");
  if (!item) {
    return;
  }
  const index = Number(item.dataset.index);
  if (!Number.isNaN(index)) {
    selectBox(index);
  }
}

function onClassChanged() {
  const classId = Number(el.classSelect.value);
  if (Number.isNaN(classId)) {
    return;
  }
  state.defaultClassId = classId;
  if (state.selectedIndex >= 0) {
    pushHistoryForCurrent();
    state.boxes[state.selectedIndex].class_id = classId;
    updateCacheForCurrent(true);
    refreshBoxList();
    render();
  }
}

function canUseGlobalHotkeys() {
  if (el.pickerModal && !el.pickerModal.classList.contains("hidden")) {
    return false;
  }
  if (el.classContextMenu && !el.classContextMenu.classList.contains("hidden")) {
    return false;
  }
  const active = document.activeElement;
  if (!active) {
    return true;
  }
  if (active.isContentEditable) {
    return false;
  }
  if (active.tagName === "TEXTAREA") {
    return false;
  }
  if (active.tagName === "INPUT") {
    const inputType = String(active.type || "").toLowerCase();
    return !["text", "search", "url", "email", "number", "password", "tel"].includes(inputType);
  }
  return true;
}

function onKeyDown(event) {
  const key = event.key;
  const lower = key.toLowerCase();

  if (key === "Escape") {
    hideClassContextMenu();
  }

  const classMenuVisible = Boolean(
    el.classContextMenu && !el.classContextMenu.classList.contains("hidden")
  );
  if (classMenuVisible && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const searchInput = getClassMenuSearchInput();
    const activeTag = document.activeElement ? document.activeElement.tagName : "";
    const activeIsSearch = searchInput && document.activeElement === searchInput;
    if (searchInput && !activeIsSearch) {
      if (key.length === 1) {
        event.preventDefault();
        updateClassMenuSearchValue(searchInput.value + key);
        return;
      }
      if (key === "Backspace") {
        event.preventDefault();
        updateClassMenuSearchValue(searchInput.value.slice(0, -1));
        return;
      }
      if (key === "Enter") {
        event.preventDefault();
        chooseFirstVisibleClassFromMenu();
        return;
      }
      if (activeTag !== "INPUT" && activeTag !== "TEXTAREA") {
        searchInput.focus();
      }
    }
  }

  if (event.ctrlKey && lower === "s" && event.shiftKey) {
    event.preventDefault();
    saveAllDirty();
    return;
  }
  if (event.ctrlKey && lower === "s") {
    event.preventDefault();
    saveCurrentImage();
    return;
  }
  if (event.ctrlKey && lower === "c") {
    if (!canUseGlobalHotkeys()) {
      return;
    }
    event.preventDefault();
    if (copySelectedBoxToClipboard()) {
      setStatus("Box copied. Press Ctrl+V to paste.");
    } else {
      setStatus("Select a box first.");
    }
    return;
  }
  if (event.ctrlKey && lower === "v") {
    if (!canUseGlobalHotkeys()) {
      return;
    }
    event.preventDefault();
    const pastedCount = pasteBoxesFromClipboard();
    if (pastedCount > 0) {
      setStatus(`Pasted ${pastedCount} box.`);
    } else {
      setStatus("Clipboard is empty.");
    }
    return;
  }
  if (event.ctrlKey && lower === "z") {
    if (!canUseGlobalHotkeys()) {
      return;
    }
    event.preventDefault();
    if (undoCurrent()) {
      setStatus("Undo applied.");
    } else {
      setStatus("Nothing to undo.");
    }
    return;
  }
  if (!canUseGlobalHotkeys()) {
    return;
  }

  if (key === "Delete" || key === "Backspace") {
    event.preventDefault();
    deleteSelectedBox();
    return;
  }
  if (lower === "p") {
    event.preventDefault();
    predictCurrentImage();
    return;
  }
  if (key === "+" || key === "=") {
    event.preventDefault();
    zoomBy(1.12);
    return;
  }
  if (key === "-" || key === "_") {
    event.preventDefault();
    zoomBy(0.88);
    return;
  }
  if (key === "0") {
    event.preventDefault();
    zoomFit();
    return;
  }

  if (event.shiftKey && state.selectedIndex >= 0) {
    const delta = event.altKey ? 10 : 2;
    if (key === "ArrowLeft") {
      event.preventDefault();
      moveSelectedBox(-delta, 0);
      return;
    }
    if (key === "ArrowRight") {
      event.preventDefault();
      moveSelectedBox(delta, 0);
      return;
    }
    if (key === "ArrowUp") {
      event.preventDefault();
      moveSelectedBox(0, -delta);
      return;
    }
    if (key === "ArrowDown") {
      event.preventDefault();
      moveSelectedBox(0, delta);
      return;
    }
  }

  if (key === "ArrowLeft") {
    event.preventDefault();
    loadImageByIndex(state.imageIndex - 1);
    return;
  }
  if (key === "ArrowRight") {
    event.preventDefault();
    loadImageByIndex(state.imageIndex + 1);
  }
}

function setPickerVisible(visible) {
  el.pickerModal.classList.toggle("hidden", !visible);
}

function selectPickerPath(path) {
  if (!state.picker.target) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  config.input.value = path;
  closePicker();
}

function renderPickerList(data) {
  el.pickerList.innerHTML = "";
  el.pickerPath.textContent = data.current_path || "Roots";
  state.picker.currentPath = data.current_path && data.current_path !== "Roots" ? data.current_path : null;
  state.picker.parentPath = data.parent_path || null;

  if (!state.picker.target) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  el.pickerSelectCurrentBtn.style.display = config.allowFolder ? "inline-block" : "none";
  el.pickerUpBtn.disabled = !data.parent_path;

  data.directories.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => loadPickerPath(entry.path, config.mode));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "DIR";
    const name = document.createElement("span");
    name.className = "name";
    name.appendChild(button);
    li.appendChild(name);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  data.files.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "picker-item";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.name;
    button.addEventListener("click", () => selectPickerPath(entry.path));
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = "FILE";
    const name = document.createElement("span");
    name.className = "name";
    name.appendChild(button);
    li.appendChild(name);
    li.appendChild(tag);
    el.pickerList.appendChild(li);
  });

  if (data.directories.length === 0 && data.files.length === 0) {
    const empty = document.createElement("li");
    empty.className = "picker-item";
    empty.innerHTML = "<span class=\"name\">No entries</span>";
    el.pickerList.appendChild(empty);
  }
}

async function showPickerRoots() {
  try {
    const data = await apiJson("/api/fs/roots");
    renderPickerList({
      current_path: "Roots",
      parent_path: null,
      directories: (data.roots || []).map((path) => ({ name: path, path })),
      files: [],
    });
  } catch (error) {
    setStatus(`Cannot load roots: ${error.message}`, true);
  }
}

async function loadPickerPath(path, mode) {
  try {
    const data = await apiJson(
      `/api/fs/list?path=${encodeURIComponent(path)}&mode=${encodeURIComponent(mode)}`
    );
    renderPickerList(data);
  } catch (error) {
    setStatus(`Path browse error: ${error.message}`, true);
  }
}

function closePicker() {
  state.picker.target = null;
  state.picker.currentPath = null;
  state.picker.parentPath = null;
  setPickerVisible(false);
}

function openPicker(target) {
  const config = PICKER_CONFIG[target];
  if (!config) {
    return;
  }
  state.picker.target = target;
  el.pickerTitle.textContent = config.title;
  setPickerVisible(true);

  const startPath = config.input.value.trim() || el.datasetDirInput.value.trim();
  if (startPath) {
    loadPickerPath(startPath, config.mode);
  } else {
    showPickerRoots();
  }
}

function onPickerSelectCurrent() {
  if (!state.picker.target || !state.picker.currentPath) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  if (!config.allowFolder) {
    return;
  }
  config.input.value = state.picker.currentPath;
  closePicker();
}

function onPickerUp() {
  if (!state.picker.target || !state.picker.parentPath) {
    return;
  }
  const config = PICKER_CONFIG[state.picker.target];
  loadPickerPath(state.picker.parentPath, config.mode);
}

el.loadSessionBtn.addEventListener("click", loadSession);
el.predictBtn.addEventListener("click", predictCurrentImage);
el.saveBtn.addEventListener("click", saveCurrentImage);
el.saveAllBtn.addEventListener("click", saveAllDirty);
if (el.applyOverlapBtn) {
  el.applyOverlapBtn.addEventListener("click", () => {
    const removed = cleanupCurrentOverlaps();
    setStatus(`Overlap cleanup removed ${removed} boxes.`);
  });
}
el.deleteBtn.addEventListener("click", deleteSelectedBox);
el.quickPredictBtn.addEventListener("click", predictCurrentImage);
el.quickSaveBtn.addEventListener("click", saveCurrentImage);
el.quickPrevBtn.addEventListener("click", () => loadImageByIndex(state.imageIndex - 1));
el.quickNextBtn.addEventListener("click", () => loadImageByIndex(state.imageIndex + 1));
el.prevBtn.addEventListener("click", () => loadImageByIndex(state.imageIndex - 1));
el.nextBtn.addEventListener("click", () => loadImageByIndex(state.imageIndex + 1));
el.zoomInBtn.addEventListener("click", () => zoomBy(1.12));
el.zoomOutBtn.addEventListener("click", () => zoomBy(0.88));
el.zoomFitBtn.addEventListener("click", zoomFit);

el.clearBtn.addEventListener("click", () => {
  pushHistoryForCurrent();
  state.boxes = [];
  state.selectedIndex = -1;
  hideClassContextMenu();
  updateCacheForCurrent(true);
  refreshBoxList();
  render();
});

el.imageSelect.addEventListener("change", () => {
  const index = Number(el.imageSelect.value);
  if (!Number.isNaN(index)) {
    loadImageByIndex(index);
  }
});

el.classSelect.addEventListener("change", onClassChanged);
el.boxList.addEventListener("click", onBoxListClick);

el.canvas.addEventListener("mousedown", onCanvasMouseDown);
el.canvas.addEventListener("mousemove", onCanvasMouseMove);
el.canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
el.canvas.addEventListener("contextmenu", onCanvasContextMenu);
el.canvas.addEventListener("dblclick", onCanvasDoubleClick);
window.addEventListener("mouseup", onCanvasMouseUp);
window.addEventListener("keydown", onKeyDown);
window.addEventListener("resize", resizeCanvas);

el.browseDatasetBtn.addEventListener("click", () => openPicker("dataset"));
el.browseModelBtn.addEventListener("click", () => openPicker("model"));
el.browseLabelsBtn.addEventListener("click", () => openPicker("labels"));
el.browseClassesBtn.addEventListener("click", () => openPicker("classes"));
el.pickerCloseBtn.addEventListener("click", closePicker);
el.pickerRootsBtn.addEventListener("click", showPickerRoots);
el.pickerSelectCurrentBtn.addEventListener("click", onPickerSelectCurrent);
el.pickerUpBtn.addEventListener("click", onPickerUp);
el.pickerModal.addEventListener("click", (event) => {
  if (event.target === el.pickerModal) {
    closePicker();
  }
});
document.addEventListener("mousedown", (event) => {
  if (!el.classContextMenu || el.classContextMenu.classList.contains("hidden")) {
    return;
  }
  const rect = el.classContextMenu.getBoundingClientRect();
  const insideRect =
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom;
  if (el.classContextMenu.contains(event.target) || insideRect) {
    return;
  }
  hideClassContextMenu();
});

resizeCanvas();
updateZoomLabel();
syncClassControls();
refreshBoxList();
setStatus("Enter dataset path and click Load Dataset.");
