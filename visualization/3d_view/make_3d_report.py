"""
make_3d_report.py
=================
Reads organella geometry parquets and generates a self-contained two-panel
synchronized 3D HTML viewer (report_3d.html).

Left panel  = WT     (fish 4)
Right panel = Mutant (fish 3)

Camera is shared — both panels rotate/zoom together. Per-panel stats overlays
show instance count and total volume (µm³) for each visible focal type.

Usage
-----
  python make_3d_report.py
"""

import base64
import json
import os
import struct

import numpy as np
import polars as pl
import trimesh

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ORGANELLA_DIR  = os.path.join(REPO_ROOT, "visualization", "organella")
REPORT_PARQUET = os.path.join(ORGANELLA_DIR, "report.parquet")
MESH_DIR       = os.path.join(ORGANELLA_DIR, "report_meshes")
OUT_HTML       = os.path.join(SCRIPT_DIR, "report_3d.html")

WT_FISH  = 4
MUT_FISH = 3

FOCAL_TYPES = [
    "blood",
    "hematopoietic cell",
    "cranial vasculature",
    "spinal cord",
    "pronephric distal early tubule",
]

COLORS = {
    "blood":                          "#C0392B",
    "hematopoietic cell":             "#8E44AD",
    "cranial vasculature":            "#E67E22",
    "spinal cord":                    "#27AE60",
    "pronephric distal early tubule": "#2980B9",
}

# ── mesh decoding ──────────────────────────────────────────────────────────────

def entity_key(ct: str) -> str:
    return ct.replace(" ", "_")


def decode_mesh(data: bytes):
    nV, nF   = struct.unpack_from('<II', data, 0)
    min_xyz  = np.frombuffer(data, dtype=np.float32, count=3, offset=8)
    scl_xyz  = np.frombuffer(data, dtype=np.float32, count=3, offset=20)
    verts_q  = np.frombuffer(data, dtype=np.uint16, count=nV * 3, offset=32).reshape(nV, 3)
    idx_dtype = np.uint16 if nV < 65536 else np.uint32
    faces    = np.frombuffer(data, dtype=idx_dtype, count=nF * 3, offset=32 + nV * 3 * 2).reshape(nF, 3)
    verts    = verts_q / 65535.0 * scl_xyz + min_xyz  # physical µm, ZYX
    return verts.astype(np.float32), faces.astype(np.uint32)


_UNIT_SPHERE = None
_UNIT_FACES  = None

def _get_icosphere():
    global _UNIT_SPHERE, _UNIT_FACES
    if _UNIT_SPHERE is None:
        s = trimesh.creation.icosphere(subdivisions=2)
        _UNIT_SPHERE = np.array(s.vertices, dtype=np.float32)
        _UNIT_FACES  = np.array(s.faces,    dtype=np.uint32)
    return _UNIT_SPHERE, _UNIT_FACES


def decode_ellipsoid(data: bytes):
    floats  = np.frombuffer(data, dtype=np.float32)
    centre  = floats[0:3]              # ZYX µm
    radii   = floats[3:6]              # ZYX µm
    axes    = floats[6:15].reshape(3, 3)
    unit_v, faces = _get_icosphere()
    verts = (unit_v * radii) @ axes + centre  # (N, 3) ZYX µm
    return verts.astype(np.float32), faces


def zyx_to_three(v: np.ndarray) -> np.ndarray:
    return np.column_stack([v[:, 0], -v[:, 1], v[:, 2]]).astype(np.float32)


def concat_meshes(pairs):
    """Concatenate list of (verts, faces) pairs into one (positions, indices)."""
    all_pos = []
    all_idx = []
    offset  = 0
    for verts, faces in pairs:
        all_pos.append(verts)
        all_idx.append(faces + offset)
        offset += len(verts)
    if not all_pos:
        return None
    return (
        np.concatenate(all_pos, axis=0),
        np.concatenate(all_idx, axis=0),
    )


def thin_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if len(faces) <= max_faces:
        return faces
    step = len(faces) // max_faces
    return faces[::step]


def to_js_geo(verts, faces, max_faces: int = 150_000):
    faces = thin_faces(faces, max_faces)
    pos_b64 = base64.b64encode(verts.astype(np.float32).flatten().tobytes()).decode()
    idx_b64 = base64.b64encode(faces.astype(np.uint32).flatten().tobytes()).decode()
    return {"pos_b64": pos_b64, "idx_b64": idx_b64}


# ── load geometry ──────────────────────────────────────────────────────────────

def load_geometry(fish: int) -> dict:
    parquet_path = os.path.join(MESH_DIR, str(fish), "geometry.parquet")
    cache_path   = os.path.join(MESH_DIR, str(fish), "geo_cache.json")

    # Use cache if it exists and is newer than the parquet
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(parquet_path):
        print(f"  fish {fish}  loading from cache ({cache_path})")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  fish {fish}  decoding geometry (will cache to {cache_path})")
    gdf    = pl.read_parquet(parquet_path)
    result = {}

    # Focal types — instances (mesh or ellipsoid)
    for ct in FOCAL_TYPES:
        ent  = entity_key(ct)
        rows = gdf.filter(
            (pl.col("entity_name") == ent) &
            (pl.col("row_type") == "instance") &
            pl.col("surface").is_not_null()
        )
        pairs = []
        for row in rows.iter_rows(named=True):
            data = bytes(row["surface"])
            if row["surface_kind"] == "mesh":
                v, f = decode_mesh(data)
            else:
                v, f = decode_ellipsoid(data)
            pairs.append((zyx_to_three(v), f))
        r = concat_meshes(pairs)
        if r is not None:
            nv = len(r[0])
            nf = len(r[1])
            print(f"  fish {fish}  {ct:40s}  {nv:7,} verts  {nf:6,} faces  ({len(pairs)} instances)")
            result[ct] = to_js_geo(r[0], r[1], max_faces=100_000)
        else:
            print(f"  fish {fish}  {ct:40s}  NO GEOMETRY")

    # Tissue — single whole-structure mesh (downsample to keep size manageable)
    tissue_rows = gdf.filter(
        (pl.col("entity_name") == "tissue") &
        (pl.col("row_type") == "file") &
        pl.col("surface").is_not_null()
    )
    if len(tissue_rows) > 0:
        data = bytes(tissue_rows["surface"][0])
        v, f = decode_mesh(data)
        target = 300_000
        step   = max(1, len(f) // target)
        f_ds   = f[::step]
        v3 = zyx_to_three(v)
        print(f"  fish {fish}  tissue  {len(v3):,} verts  {len(f_ds):,} faces (step {step}x from {len(f):,})")
        result["tissue"] = to_js_geo(v3, f_ds, max_faces=80_000)

    with open(cache_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"  fish {fish}  cache written → {cache_path}")

    return result


# ── load stats ─────────────────────────────────────────────────────────────────

def load_stats() -> dict:
    df   = pl.read_parquet(REPORT_PARQUET)
    rows = df.filter(pl.col("row_type") == pl.lit("entity")).filter(
        pl.col("object_id").is_in(["3", "4"])
    )
    stats = {}
    for row in rows.iter_rows(named=True):
        fish = int(row["object_id"])
        ent  = row["entity_name"]
        stats.setdefault(fish, {})[ent] = {
            "instances":  int(row["instance_count"]) if row["instance_count"] is not None else 0,
            "volume_um3": round(float(row["total_volume_um3"]), 0),
        }
    return stats


# ── HTML generation ────────────────────────────────────────────────────────────

def generate_html(wt_geo: dict, mut_geo: dict, stats: dict) -> str:
    def fmt_stats(fish_stats: dict) -> dict:
        out = {}
        for ct in FOCAL_TYPES:
            ent = entity_key(ct)
            s   = fish_stats.get(ent, {})
            out[ct] = {
                "instances":  s.get("instances", 0),
                "volume_um3": s.get("volume_um3", 0.0),
            }
        return out

    data = {
        "wt":           wt_geo,
        "mutant":       mut_geo,
        "stats": {
            "wt":     fmt_stats(stats.get(WT_FISH,  {})),
            "mutant": fmt_stats(stats.get(MUT_FISH, {})),
        },
    }

    data_json        = json.dumps(data, separators=(",", ":"))
    colors_json      = json.dumps(COLORS)
    focal_types_json = json.dumps(FOCAL_TYPES)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Zebrafish — 3D Cell Types</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#ddd;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}}
#sidebar{{width:210px;flex-shrink:0;background:#1c1c1c;border-right:1px solid #333;padding:14px 12px;overflow-y:auto;display:flex;flex-direction:column;gap:18px}}
#sidebar h2{{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:#777;padding-bottom:4px;border-bottom:1px solid #333}}
.ct-row{{display:flex;align-items:center;gap:8px;cursor:pointer;padding:3px 0;user-select:none}}
.ct-row input{{cursor:pointer;accent-color:#aaa}}
.swatch{{width:11px;height:11px;border-radius:2px;flex-shrink:0}}
.ct-label{{font-size:14px;line-height:1.3}}
.grp-row{{display:flex;align-items:center;gap:8px;font-size:14px;cursor:pointer;padding:2px 0}}
.grp-row input{{cursor:pointer;accent-color:#aaa}}
.note{{font-size:12px;color:#555;margin-top:-2px;padding-left:20px}}
#canvas-wrap{{flex:1;position:relative}}
canvas{{display:block}}
.panel-label{{position:absolute;top:10px;font-size:15px;font-weight:600;letter-spacing:.04em;pointer-events:none;padding:4px 10px;border-radius:4px;background:rgba(0,0,0,.45)}}
#lbl-wt{{left:25%;transform:translateX(-50%);color:#fff}}
#lbl-mut{{left:75%;transform:translateX(-50%);color:#fff}}
.stats-overlay{{position:absolute;bottom:14px;pointer-events:none;font-size:13px;color:#bbb;background:rgba(0,0,0,.5);border-radius:5px;padding:7px 9px;min-width:150px;line-height:1.6}}
#stats-wt{{left:10px}}
#stats-mut{{right:10px;text-align:right}}
.stats-overlay table{{border-collapse:collapse;width:100%}}
.stats-overlay td{{padding:0 4px}}
.stats-overlay .vol{{color:#888;font-size:12px}}
#info{{position:absolute;bottom:4px;left:50%;transform:translateX(-50%);font-size:12px;color:#444;pointer-events:none}}
button#reset-btn{{background:#2a2a2a;border:1px solid #444;color:#bbb;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:14px;width:100%}}
button#reset-btn:hover{{background:#333}}
#record-btn{{background:#2a2a2a;border:1px solid #444;color:#bbb;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:14px;width:100%;margin-top:8px}}
#record-btn:not(:disabled):hover{{background:#333}}
#record-btn:disabled{{opacity:.4;cursor:default}}
#record-btn.recording{{background:#7a1c1c;border-color:#a33;color:#fcc}}
input[type=range]{{width:100%;accent-color:#aaa;margin-top:4px}}
</style>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}}}
</script>
</head>
<body>

<div id="sidebar">
  <div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
      <h2 style="margin-bottom:0">Cell types</h2>
      <label style="display:flex;align-items:center;gap:5px;font-size:12px;color:#888;cursor:pointer">
        <input type="checkbox" id="show-legend" checked>
        <span>Show</span>
      </label>
    </div>
    <div id="ct-toggles"></div>
  </div>
  <div>
    <h2 style="margin-bottom:8px">Context</h2>
    <label class="grp-row">
      <input type="checkbox" id="show-tissue" checked>
      <span style="color:#888">Tissue (all)</span>
    </label>
    <div class="note">faint gray backdrop</div>
  </div>
  <div>
    <h2 style="margin-bottom:8px">Camera</h2>
    <button id="reset-btn">Reset view</button>
    <div style="margin-top:8px;font-size:12px;color:#555;line-height:1.6">
      Drag — orbit · Scroll — zoom<br>Right-drag — pan
    </div>
  </div>
  <div>
    <h2 style="margin-bottom:8px">Rotation</h2>
    <label class="grp-row">
      <input type="checkbox" id="auto-rotate">
      <span>Auto-rotate</span>
    </label>
    <div style="margin-top:10px">
      <div style="font-size:12px;color:#666;margin-bottom:2px">Tilt &mdash; <span id="tilt-deg">0</span>&deg;</div>
      <input type="range" id="tilt-slider" min="0" max="90" step="1" value="0">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#555;margin-top:1px"><span>Horizontal</span><span>Roll</span></div>
    </div>
    <div style="margin-top:10px">
      <div style="font-size:12px;color:#666;margin-bottom:2px">Speed &mdash; <span id="orbit-secs">30</span>s / orbit</div>
      <input type="range" id="speed-slider" min="0.3" max="10" step="0.1" value="1">
    </div>
    <button id="record-btn" disabled>Record</button>
    <div class="note" style="margin-top:4px">starts 1s after click · ends 1s before clicking Stop</div>
    <label class="grp-row" style="margin-top:6px">
      <input type="checkbox" id="stats-in-rec" checked>
      <span>Include stats in recording</span>
    </label>
  </div>
</div>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div class="panel-label" id="lbl-wt">WT - fish {WT_FISH}</div>
  <div class="panel-label" id="lbl-mut">Mutant - fish {MUT_FISH}</div>
  <div class="stats-overlay" id="stats-wt"></div>
  <div class="stats-overlay" id="stats-mut"></div>
  <div id="info">synchronized camera · drag to orbit</div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA        = {data_json};
const COLORS      = {colors_json};
const FOCAL_TYPES = {focal_types_json};

// ── scenes ──────────────────────────────────────────────────────────────────
const sceneWT  = new THREE.Scene();
const sceneMut = new THREE.Scene();

function addLights(sc) {{
  sc.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(800, 1200, 600);
  sc.add(sun);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4);
  fill.position.set(-600, -400, -800);
  sc.add(fill);
}}
addLights(sceneWT);
addLights(sceneMut);

// ── renderer / camera / controls ─────────────────────────────────────────────
const canvas     = document.getElementById('c');
const compCanvas = document.createElement('canvas');
const compCtx    = compCanvas.getContext('2d');
const wrap     = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x111111);

const camera   = new THREE.PerspectiveCamera(45, 1, 1, 50000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping    = true;
controls.dampingFactor    = 0.08;

// ── geometry helpers ─────────────────────────────────────────────────────────
function b64ToBuffer(b64, TypedArray) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new TypedArray(buf);
}}

function makeGeo(data) {{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position',
    new THREE.BufferAttribute(b64ToBuffer(data.pos_b64, Float32Array), 3));
  geo.setIndex(new THREE.BufferAttribute(b64ToBuffer(data.idx_b64, Uint32Array), 1));
  geo.computeVertexNormals();
  return geo;
}}

// ── build meshes ─────────────────────────────────────────────────────────────
const focalMeshes = {{  // focalMeshes[ct] = [wtMesh, mutMesh]
}};
const allVerts = [];

function collectVerts(geo) {{
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++)
    allVerts.push(pos.getX(i), pos.getY(i), pos.getZ(i));
}}

for (const ct of FOCAL_TYPES) {{
  const color = COLORS[ct] ?? '#888888';
  const pair  = [];
  for (const [grp, scene] of [['wt', sceneWT], ['mutant', sceneMut]]) {{
    const d = DATA[grp]?.[ct];
    if (!d) {{ pair.push(null); continue; }}
    const geo = makeGeo(d);
    collectVerts(geo);
    const mat  = new THREE.MeshPhongMaterial({{
      color, side: THREE.DoubleSide, shininess: 35,
      specular: new THREE.Color(0x222222),
    }});
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    pair.push(mesh);
  }}
  focalMeshes[ct] = pair;
}}

// tissue context
const tissueMat = new THREE.MeshPhongMaterial({{
  color: 0x888888, opacity: 0.07, transparent: true,
  side: THREE.DoubleSide, shininess: 0, depthWrite: false,
}});
const tissueWT  = DATA.wt?.tissue  ? new THREE.Mesh(makeGeo(DATA.wt.tissue),     tissueMat.clone()) : null;
const tissueMut = DATA.mutant?.tissue ? new THREE.Mesh(makeGeo(DATA.mutant.tissue), tissueMat.clone()) : null;
if (tissueWT)  sceneWT.add(tissueWT);
if (tissueMut) sceneMut.add(tissueMut);

// ── camera initial position ──────────────────────────────────────────────────
const bbox   = new THREE.Box3().setFromArray(allVerts);
const centre = bbox.getCenter(new THREE.Vector3());
const size   = bbox.getSize(new THREE.Vector3()).length();
controls.target.copy(centre);
camera.position.copy(centre).add(new THREE.Vector3(0, size * 0.15, size * 0.9));
camera.near = size * 0.001;
camera.far  = size * 10;
camera.updateProjectionMatrix();
const INIT_POS = camera.position.clone();
const INIT_TGT = controls.target.clone();

// ── stats overlay ─────────────────────────────────────────────────────────────
function fmtVol(v) {{
  if (v >= 1e6)  return `>${{Math.round(v/1e6)}}M µm³`;
  if (v >= 1000) return `${{Math.round(v/1000)}}k µm³`;
  return `${{Math.round(v)}} µm³`;
}}

function updateStats() {{
  for (const [id, grp] of [['stats-wt','wt'],['stats-mut','mutant']]) {{
    const div = document.getElementById(id);
    const rows = [];
    for (const ct of FOCAL_TYPES) {{
      const [wtM, mutM] = focalMeshes[ct];
      const m = grp === 'wt' ? wtM : mutM;
      if (!m || !m.visible) continue;
      const s = DATA.stats[grp][ct];
      const color = COLORS[ct];
      rows.push(`<tr>
        <td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${{color}};margin-right:4px"></span>${{ct}}</td>
        <td style="text-align:right">${{s.instances}} inst</td>
      </tr><tr><td colspan="2" class="vol" style="padding-bottom:2px;${{grp==='mutant'?'text-align:right':''}}">${{fmtVol(s.volume_um3)}}</td></tr>`);
    }}
    div.innerHTML = rows.length
      ? `<table>${{rows.join('')}}</table>`
      : '<span style="color:#555">nothing visible</span>';
  }}
}}

// ── sidebar ───────────────────────────────────────────────────────────────────
const ctContainer = document.getElementById('ct-toggles');
for (const ct of FOCAL_TYPES) {{
  const row = document.createElement('label');
  row.className = 'ct-row';
  row.innerHTML = `<input type="checkbox" checked data-ct="${{ct}}">
    <span class="swatch" style="background:${{COLORS[ct]??'#888'}}"></span>
    <span class="ct-label">${{ct}}</span>`;
  ctContainer.appendChild(row);
}}

function applyVisibility() {{
  for (const ct of FOCAL_TYPES) {{
    const cb   = document.querySelector(`[data-ct="${{ct}}"]`);
    const on   = cb?.checked ?? true;
    const [wm, mm] = focalMeshes[ct];
    if (wm)  wm.visible  = on;
    if (mm)  mm.visible  = on;
  }}
  updateStats();
}}

ctContainer.addEventListener('change', applyVisibility);

document.getElementById('show-tissue').addEventListener('change', e => {{
  if (tissueWT)  tissueWT.visible  = e.target.checked;
  if (tissueMut) tissueMut.visible = e.target.checked;
}});

document.getElementById('reset-btn').addEventListener('click', () => {{
  camera.position.copy(INIT_POS);
  controls.target.copy(INIT_TGT);
  controls.update();
}});

// ── auto-rotate + recording ──────────────────────────────────────────────────
const autoRotateCb  = document.getElementById('auto-rotate');
const tiltSlider    = document.getElementById('tilt-slider');
const tiltDegSpan   = document.getElementById('tilt-deg');
const speedSlider   = document.getElementById('speed-slider');
const orbitSecsSpan = document.getElementById('orbit-secs');
const recordBtn     = document.getElementById('record-btn');
const showLegendCb  = document.getElementById('show-legend');
const statsInRecCb  = document.getElementById('stats-in-rec');

let isAutoRotating = false;
let rotAngle = 0;
let lastRotTime = null;

autoRotateCb.addEventListener('change', () => {{
  isAutoRotating = autoRotateCb.checked;
  recordBtn.disabled = !isAutoRotating;
  lastRotTime = null;
  if (!isAutoRotating) stopRecording();
}});

tiltSlider.addEventListener('input', () => {{
  tiltDegSpan.textContent = tiltSlider.value;
}});

speedSlider.addEventListener('input', () => {{
  orbitSecsSpan.textContent = Math.round(30 / parseFloat(speedSlider.value));
}});

showLegendCb.addEventListener('change', () => {{
  document.getElementById('ct-toggles').style.display = showLegendCb.checked ? '' : 'none';
}});

function stepAutoRotation() {{
  const now = performance.now();
  if (lastRotTime === null) {{ lastRotTime = now; return; }}
  const dt = Math.min((now - lastRotTime) / 1000, 0.05);
  lastRotTime = now;
  const speed = parseFloat(speedSlider.value);
  const dr = speed * 2 * Math.PI / 30 * dt;
  rotAngle += dr;
  const tiltRad = parseFloat(tiltSlider.value) * Math.PI / 180;
  const axis = new THREE.Vector3(Math.sin(tiltRad), Math.cos(tiltRad), 0);
  const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
  offset.applyQuaternion(new THREE.Quaternion().setFromAxisAngle(axis, dr));
  camera.position.copy(controls.target).add(offset);
}}

let mediaRecorder = null, recordChunks = [], chunkTimes = [];

function setRecordingUI(active) {{
  if (active) {{
    recordBtn.classList.add('recording');
    recordBtn.textContent = 'Stop recording';
    recordBtn.disabled = false;
  }} else {{
    recordBtn.classList.remove('recording');
    recordBtn.textContent = 'Record';
    recordBtn.disabled = !autoRotateCb.checked;
  }}
}}

function stopRecording() {{
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}}

recordBtn.addEventListener('click', () => {{
  if (mediaRecorder) {{
    recordBtn.disabled = true;
    recordBtn.textContent = 'Stopping…';
    stopRecording();
    return;
  }}
  recordBtn.disabled = true;
  recordBtn.textContent = 'Starting…';
  setTimeout(() => {{
    recordChunks = []; chunkTimes = [];
    const stream   = compCanvas.captureStream(30);
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? 'video/webm;codecs=vp9' : 'video/webm';
    mediaRecorder = new MediaRecorder(stream, {{ mimeType, videoBitsPerSecond: 10e6 }});
    mediaRecorder.ondataavailable = e => {{
      if (e.data.size > 0) {{ recordChunks.push(e.data); chunkTimes.push(performance.now()); }}
    }};
    mediaRecorder.onstop = () => {{
      const cutoff = performance.now() - 1000;
      const keep = recordChunks.filter((_, i) => chunkTimes[i] <= cutoff);
      const blob = new Blob(keep.length > 1 ? keep : recordChunks, {{ type: 'video/webm' }});
      const a = Object.assign(document.createElement('a'), {{
        href: URL.createObjectURL(blob), download: 'zebrafish_rotation.webm'
      }});
      a.click();
      mediaRecorder = null;
      setRecordingUI(false);
    }};
    mediaRecorder.start(100);
    setRecordingUI(true);
  }}, 1000);
}});

// ── resize ────────────────────────────────────────────────────────────────────
function resize() {{
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  document.getElementById('lbl-wt').style.left  = (w * 0.25) + 'px';
  document.getElementById('lbl-mut').style.left = (w * 0.75) + 'px';
}}
new ResizeObserver(resize).observe(wrap);
resize();

// ── render loop (scissor split) ───────────────────────────────────────────────
function drawComposite() {{
  if (compCanvas.width  !== canvas.width)  compCanvas.width  = canvas.width;
  if (compCanvas.height !== canvas.height) compCanvas.height = canvas.height;
  compCtx.drawImage(canvas, 0, 0);
  const w = compCanvas.width;
  compCtx.font = 'bold 15px system-ui,sans-serif';
  compCtx.textAlign = 'center';
  compCtx.textBaseline = 'top';
  [['WT - fish {WT_FISH}', w * 0.25], ['Mutant - fish {MUT_FISH}', w * 0.75]].forEach(([lbl, x]) => {{
    const tw = compCtx.measureText(lbl).width;
    compCtx.fillStyle = 'rgba(0,0,0,0.45)';
    compCtx.fillRect(x - tw / 2 - 10, 6, tw + 20, 26);
    compCtx.fillStyle = '#ffffff';
    compCtx.fillText(lbl, x, 10);
  }});
  if (statsInRecCb && statsInRecCb.checked) {{
    const h = compCanvas.height;
    [['stats-wt', false], ['stats-mut', true]].forEach(([id, isRight]) => {{
      const el = document.getElementById(id);
      if (!el || !el.innerHTML.trim()) return;
      const rows = [];
      el.querySelectorAll('tr').forEach(tr => {{
        const tds = tr.querySelectorAll('td');
        if (tds.length >= 2) rows.push([tds[0].textContent.trim(), tds[1].textContent.trim()]);
        else if (tds.length === 1) rows.push([tds[0].textContent.trim(), '']);
      }});
      if (!rows.length) return;
      const lineH = 18, pad = 8, colW = 110;
      const boxW = colW * 2 + pad * 2;
      const boxH = rows.length * lineH + pad * 2;
      const x = isRight ? w - boxW - 10 : 10;
      const y = h - boxH - 14;
      compCtx.fillStyle = 'rgba(0,0,0,0.5)';
      compCtx.beginPath();
      compCtx.roundRect(x, y, boxW, boxH, 5);
      compCtx.fill();
      compCtx.font = '12px system-ui,sans-serif';
      compCtx.textBaseline = 'top';
      rows.forEach(([left, right], i) => {{
        const ty = y + pad + i * lineH;
        compCtx.fillStyle = left.includes('µ') ? '#888' : '#bbb';
        compCtx.textAlign = 'left';
        compCtx.fillText(left, x + pad, ty);
        compCtx.textAlign = 'right';
        compCtx.fillText(right, x + boxW - pad, ty);
      }});
    }});
  }}
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  if (isAutoRotating) stepAutoRotation();
  const W  = renderer.domElement.width;
  const H  = renderer.domElement.height;
  const hw = Math.floor(W / 2);
  renderer.setScissorTest(true);

  // Left: WT
  renderer.setViewport(0, 0, hw, H);
  renderer.setScissor(0, 0, hw, H);
  camera.aspect = hw / H;
  camera.updateProjectionMatrix();
  renderer.render(sceneWT, camera);

  // Right: Mutant
  renderer.setViewport(hw, 0, W - hw, H);
  renderer.setScissor(hw, 0, W - hw, H);
  renderer.render(sceneMut, camera);
  if (mediaRecorder) drawComposite();
}}
animate();
updateStats();
</script>
</body>
</html>
"""


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading WT geometry  (fish {WT_FISH})…")
    wt_geo  = load_geometry(WT_FISH)
    print(f"Loading Mutant geometry (fish {MUT_FISH})…")
    mut_geo = load_geometry(MUT_FISH)
    print("Loading stats…")
    stats   = load_stats()

    print("Generating HTML…")
    html = generate_html(wt_geo, mut_geo, stats)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(OUT_HTML) / 1e6
    print(f"\nWritten: {OUT_HTML}  ({size_mb:.1f} MB)")
    print(f"Open in browser:  file://{OUT_HTML}")


if __name__ == "__main__":
    main()
