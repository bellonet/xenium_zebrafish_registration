"""
make_3d_report.py
=================
Reads the .ply meshes from visualization/meshes/ and generates a self-contained
interactive 3D HTML viewer (report_3d.html) using Three.js.

All mesh data is embedded inline — no server needed, open the file directly.

Scene:
  • WT and Mutant meshes overlaid in the same registered space
  • WT  = solid (opacity 1.0)
  • Mutant = semi-transparent (opacity 0.55) so WT shows through
  • Orbit controls — drag to rotate, scroll to zoom
  • Left panel — toggle each cell type and each group on/off

Coordinate mapping (image → Three.js):
  Image voxels have physical size Z=10 µm, XY=1.7 µm.
  marching_cubes returns vertices as (z_phys, y_phys, x_phys).
  Mapped to Three.js (Y-up):
    three_x =  vertex_z   (slice axis = A-P axis of fish → horizontal)
    three_y = -vertex_y   (flip: image Y increases downward)
    three_z =  vertex_x   (bilateral axis → depth)

Usage
-----
  python make_3d_report.py
"""

import json
import os
import re

import numpy as np
import trimesh

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR   = os.path.join(SCRIPT_DIR, "meshes")
OUT_HTML   = os.path.join(SCRIPT_DIR, "report_3d.html")

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

# ── helpers ────────────────────────────────────────────────────────────────────

def safe_filename(cell_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cell_type.lower()).strip("_")


def load_mesh(group: str, cell_type: str) -> dict | None:
    fname = f"{group}_{safe_filename(cell_type)}.ply"
    path  = os.path.join(MESH_DIR, fname)
    if not os.path.exists(path):
        print(f"  MISSING: {fname}")
        return None
    mesh  = trimesh.load(path)
    verts = mesh.vertices  # (N, 3) as (z_phys, y_phys, x_phys) in µm
    # Remap to Three.js Y-up: long axis horizontal, dorsal-ventral vertical
    three = np.column_stack([verts[:, 0], -verts[:, 1], verts[:, 2]])
    return {
        "positions": three.flatten().tolist(),
        "indices":   mesh.faces.flatten().tolist(),
    }


# ── HTML generation ────────────────────────────────────────────────────────────

def generate_html(mesh_data: dict) -> str:
    mesh_json        = json.dumps(mesh_data)
    colors_json      = json.dumps(COLORS)
    focal_types_json = json.dumps(FOCAL_TYPES)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Zebrafish — 3D Cell Types</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #111;
  color: #ddd;
  font-family: system-ui, sans-serif;
  display: flex;
  height: 100vh;
  overflow: hidden;
}}

/* ── sidebar ── */
#sidebar {{
  width: 210px;
  flex-shrink: 0;
  background: #1c1c1c;
  border-right: 1px solid #333;
  padding: 14px 12px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
#sidebar h2 {{
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: #888;
  padding-bottom: 4px;
  border-bottom: 1px solid #333;
}}
.ct-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 3px 0;
  user-select: none;
}}
.ct-row input {{ cursor: pointer; accent-color: #aaa; }}
.swatch {{
  width: 12px; height: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}}
.ct-label {{ font-size: 12px; line-height: 1.3; }}
.group-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  cursor: pointer;
  padding: 2px 0;
}}
.group-row input {{ cursor: pointer; accent-color: #aaa; }}
.wt-badge  {{ color: #7ab4f5; font-weight: 600; }}
.mut-badge {{ color: #f5a07a; font-weight: 600; }}
.opacity-note {{ font-size: 11px; color: #666; margin-top: -4px; }}

/* ── canvas ── */
#canvas-wrap {{
  flex: 1;
  position: relative;
}}
canvas {{ display: block; }}
#info {{
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: #555;
  pointer-events: none;
}}
</style>

<script type="importmap">
{{
  "imports": {{
    "three":          "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/":  "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
  }}
}}
</script>
</head>

<body>

<!-- ── sidebar ── -->
<div id="sidebar">
  <div>
    <h2 style="margin-bottom:10px">Cell types</h2>
    <div id="ct-toggles"></div>
  </div>
  <div>
    <h2 style="margin-bottom:10px">Groups</h2>
    <label class="group-row">
      <input type="checkbox" id="show-wt" checked>
      <span class="wt-badge">WT</span> fish 4
    </label>
    <div class="opacity-note" style="margin-left:24px">solid</div>
    <label class="group-row" style="margin-top:6px">
      <input type="checkbox" id="show-mut" checked>
      <span class="mut-badge">Mutant</span> fish 3
    </label>
    <div class="opacity-note" style="margin-left:24px">semi-transparent</div>
    <label class="group-row" style="margin-top:10px">
      <input type="checkbox" id="show-ctx" checked>
      <span style="color:#888">Tissue context</span>
    </label>
    <div class="opacity-note" style="margin-left:24px">gray, very faint</div>
  </div>
  <div>
    <h2 style="margin-bottom:8px">Camera</h2>
    <button id="reset-btn" style="
      background:#2a2a2a;border:1px solid #444;color:#bbb;
      padding:5px 10px;border-radius:4px;cursor:pointer;font-size:12px;width:100%">
      Reset view
    </button>
    <div style="margin-top:8px;font-size:11px;color:#555;line-height:1.5">
      Drag — orbit<br>Scroll — zoom<br>Right-drag — pan
    </div>
  </div>
</div>

<!-- ── 3D canvas ── -->
<div id="canvas-wrap">
  <canvas id="c"></canvas>
  <div id="info">WT (solid) · Mutant (transparent) · same registered space</div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

// ── embedded mesh data ──────────────────────────────────────────────────────
const MESH_DATA   = {mesh_json};
const COLORS      = {colors_json};
const FOCAL_TYPES = {focal_types_json};

// ── scene setup ─────────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const wrap     = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = false;
renderer.setClearColor(0x111111);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 1, 50000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(800, 1200, 600);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.4);
fill.position.set(-600, -400, -800);
scene.add(fill);

// ── build geometries ─────────────────────────────────────────────────────────
const meshes = {{ wt: {{}}, mutant: {{}} }};

// Groups for side-by-side layout — offset along Z after we know the size
const wtGroup  = new THREE.Group();
const mutGroup = new THREE.Group();
scene.add(wtGroup);
scene.add(mutGroup);

function makeGeo(data) {{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(data.positions), 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(data.indices), 1));
  geo.computeVertexNormals();
  return geo;
}}

// Context tissue — gray backdrop per group
const ctxMat = new THREE.MeshPhongMaterial({{
  color: 0x888888, opacity: 0.08, transparent: true,
  side: THREE.DoubleSide, shininess: 0, depthWrite: false,
}});
let contextMeshWT = null, contextMeshMut = null;
if (MESH_DATA.context_wt) {{
  contextMeshWT = new THREE.Mesh(makeGeo(MESH_DATA.context_wt), ctxMat.clone());
  wtGroup.add(contextMeshWT);
}}
if (MESH_DATA.context_mutant) {{
  contextMeshMut = new THREE.Mesh(makeGeo(MESH_DATA.context_mutant), ctxMat.clone());
  mutGroup.add(contextMeshMut);
}}

function buildMesh(group, cellType) {{
  const data = MESH_DATA[group]?.[cellType];
  if (!data) return null;
  const geo   = makeGeo(data);
  const color = COLORS[cellType] ?? '#888888';
  const mat   = new THREE.MeshPhongMaterial({{
    color, opacity: 0.90, transparent: false,
    side: THREE.DoubleSide, shininess: 35,
    specular: new THREE.Color(0x222222),
  }});
  return new THREE.Mesh(geo, mat);
}}

// Build meshes, collect verts for bounding box
const allVerts = [];
for (const [group, grpObj] of [['wt', wtGroup], ['mutant', mutGroup]]) {{
  for (const ct of FOCAL_TYPES) {{
    const m = buildMesh(group, ct);
    if (!m) continue;
    meshes[group][ct] = m;
    grpObj.add(m);
    const pos = m.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++)
      allVerts.push(pos.getX(i), pos.getY(i), pos.getZ(i));
  }}
}}

// Compute per-group Z extent to set side-by-side offset
const rawBox  = new THREE.Box3().setFromArray(allVerts);
const zExtent = rawBox.max.z - rawBox.min.z;
const GAP     = zExtent * 0.3;   // 30% gap between the two fish
wtGroup.position.z  = -(zExtent / 2 + GAP / 2);
mutGroup.position.z =  (zExtent / 2 + GAP / 2);

// Centre camera on full scene (both groups after offset)
const sceneBbox = new THREE.Box3().expandByObject(wtGroup).expandByObject(mutGroup);
const centre = sceneBbox.getCenter(new THREE.Vector3());
const size   = sceneBbox.getSize(new THREE.Vector3()).length();
controls.target.copy(centre);
camera.position.copy(centre).add(new THREE.Vector3(0, size * 0.15, size * 0.9));
camera.near = size * 0.001;
camera.far  = size * 10;
camera.updateProjectionMatrix();
const INITIAL_CAM = camera.position.clone();
const INITIAL_TGT = controls.target.clone();

// ── sidebar UI ──────────────────────────────────────────────────────────────
const ctContainer = document.getElementById('ct-toggles');
FOCAL_TYPES.forEach(ct => {{
  const row = document.createElement('label');
  row.className = 'ct-row';
  row.innerHTML = `
    <input type="checkbox" checked data-ct="${{ct}}">
    <span class="swatch" style="background:${{COLORS[ct] ?? '#888'}}"></span>
    <span class="ct-label">${{ct}}</span>`;
  ctContainer.appendChild(row);
}});

function updateVisibility() {{
  const showWT  = document.getElementById('show-wt').checked;
  const showMut = document.getElementById('show-mut').checked;
  document.querySelectorAll('[data-ct]').forEach(cb => {{
    const ct      = cb.dataset.ct;
    const ctOn    = cb.checked;
    if (meshes.wt[ct])     meshes.wt[ct].visible     = ctOn && showWT;
    if (meshes.mutant[ct]) meshes.mutant[ct].visible  = ctOn && showMut;
  }});
}}

document.getElementById('ct-toggles').addEventListener('change', updateVisibility);
document.getElementById('show-wt').addEventListener('change', updateVisibility);
document.getElementById('show-mut').addEventListener('change', updateVisibility);
document.getElementById('show-ctx').addEventListener('change', e => {{
  if (contextMeshWT)  contextMeshWT.visible  = e.target.checked;
  if (contextMeshMut) contextMeshMut.visible = e.target.checked;
}});

document.getElementById('reset-btn').addEventListener('click', () => {{
  camera.position.copy(INITIAL_CAM);
  controls.target.copy(INITIAL_TGT);
  controls.update();
}});

// ── resize ───────────────────────────────────────────────────────────────────
function resize() {{
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
new ResizeObserver(resize).observe(wrap);
resize();

// ── render loop ──────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


# ── main ───────────────────────────────────────────────────────────────────────

def load_context_mesh(group: str) -> dict | None:
    path = os.path.join(MESH_DIR, f"context_{group}.ply")
    if not os.path.exists(path):
        print(f"  context_{group}.ply not found — skipping")
        return None
    mesh  = trimesh.load(path)
    verts = mesh.vertices
    three = np.column_stack([verts[:, 0], -verts[:, 1], verts[:, 2]])
    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    print(f"  context {group}: {n_v:,} verts  {n_f:,} faces")
    return {
        "positions": three.flatten().tolist(),
        "indices":   mesh.faces.flatten().tolist(),
    }


def main() -> None:
    print("Loading meshes…")
    mesh_data: dict = {"wt": {}, "mutant": {}}
    for group in ("wt", "mutant"):
        for ct in FOCAL_TYPES:
            data = load_mesh(group, ct)
            if data:
                mesh_data[group][ct] = data
                n_v = len(data["positions"]) // 3
                n_f = len(data["indices"]) // 3
                print(f"  {group:7s} {ct:40s} {n_v:6,} verts  {n_f:6,} faces")

    for grp in ("wt", "mutant"):
        ctx = load_context_mesh(grp)
        if ctx:
            mesh_data[f"context_{grp}"] = ctx

    html = generate_html(mesh_data)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"\nWritten: {OUT_HTML}")
    print(f"Open in browser:  file://{OUT_HTML}")


if __name__ == "__main__":
    main()
