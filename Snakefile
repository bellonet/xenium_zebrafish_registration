"""
Snakemake workflow for the zebrafish registration pipeline (scripts 1–4).
Script 5 (apply_registration) is run manually — see README.

Usage (from the repo root):
    snakemake -j 6          # run pipeline, up to 6 parallel jobs
    snakemake -j 6 -n       # dry-run: print what would be done
    snakemake -j 6 -p       # print shell commands as they run
    snakemake -j 6 --rerun-incomplete   # recover from interrupted jobs

Logs go to ../analysis/logs/.
"""

# ── configuration ──────────────────────────────────────────────────────────────

FISH = list(range(1, 7))

SCRIPT3_EXPERIMENTS = [
    "tissue_mask",
    "gene_composite",
    "dapi_blend",
    "multi_metric",
]

SCRIPT4_DRIVING = ["dapi", "dapi_blend"]
SCRIPT4_STAGES  = ["rigid", "rigid_affine"]

INPUT_FOLDERS = [
    "output-XETG00046__0038328__Region_1__20250717__075022",
    "output-XETG00046__0043921__Region_1__20250620__084504",
    "output-XETG00046__0044004__Region_1__20250620__084505",
]

PYTHON   = "python3"
DATA     = "../data"
ANALYSIS = "../analysis"

# Constrain wildcards so Snakemake can unambiguously parse underscore-separated
# experiment names like dapi_blend_rigid_affine_z10.
wildcard_constraints:
    driving    = "|".join(sorted(SCRIPT4_DRIVING, key=len, reverse=True)),  # longest first
    stages     = "|".join(sorted(SCRIPT4_STAGES,  key=len, reverse=True)),
    experiment = "|".join(SCRIPT3_EXPERIMENTS),
    fish       = r"\d+",


# ── target ─────────────────────────────────────────────────────────────────────

rule all:
    input:
        expand(
            f"{ANALYSIS}/4_registered/{{driving}}_{{stages}}_z10/{{fish}}/c0.tif",
            driving=SCRIPT4_DRIVING,
            stages=SCRIPT4_STAGES,
            fish=FISH,
        ),
        f"{ANALYSIS}/4_registered/evaluation.csv",
        f"{ANALYSIS}/4_registered/evaluation_per_fish.csv",


# ── script 1: crop and tag 2D slices ───────────────────────────────────────────

rule script1:
    """
    Extract, detect, and unify per-fish 2D slices across all three Xenium runs.
    Runs once; outputs the unified individual_fish_2d directory used by script 2.
    """
    input:
        expand(
            f"{DATA}/{{folder}}/morphology_focus/morphology_focus_0000.ome.tif",
            folder=INPUT_FOLDERS,
        )
    output:
        directory(f"{ANALYSIS}/1_detection/individual_fish_2d")
    log:
        f"{ANALYSIS}/logs/script1.log"
    shell:
        "{PYTHON} 1_crop_and_tag_2d_slices.py > {log} 2>&1"


# ── script 2: per-fish 2D rigid registration ───────────────────────────────────

rule script2:
    """
    Rigid 2D registration for one fish: register → segmentation → stack → per_gene.
    Six jobs run in parallel (one per fish).
    Sentinel: rigid_3d_c0.tif produced by the stack step.
    """
    input:
        f"{ANALYSIS}/1_detection/individual_fish_2d"
    output:
        f"{ANALYSIS}/2_registered/{{fish}}/rigid_3d_c0.tif"
    log:
        f"{ANALYSIS}/logs/script2_fish{{fish}}.log"
    shell:
        "{PYTHON} 2_rigid_registration.py --fish {wildcards.fish} > {log} 2>&1"


# ── script 3: per-slice correction experiments ─────────────────────────────────

rule script3:
    """
    Second-pass 2D correction for one fish × one experiment combination.
    Up to 24 jobs run in parallel (6 fish × 4 experiments).
    Sentinel: c0_3d.tif produced after stacking the corrected slices.
    """
    input:
        f"{ANALYSIS}/2_registered/{{fish}}/rigid_3d_c0.tif"
    output:
        f"{ANALYSIS}/3_improved_registration/{{experiment}}/{{fish}}/c0_3d.tif"
    log:
        f"{ANALYSIS}/logs/script3_{{experiment}}_fish{{fish}}.log"
    shell:
        "{PYTHON} 3_registration_experiments.py "
        "--fish {wildcards.fish} --experiments {wildcards.experiment} "
        "> {log} 2>&1"


# ── script 4: 3D cross-fish registration ───────────────────────────────────────

rule script4_register:
    """
    3D cross-fish rigid/affine registration for one driving × stages combination.
    Six combinations run in parallel; each job registers all six fish.
    Depends on dapi_blend from script 3 (hardcoded in script 4 as the driving source)
    and rigid_3d stacks from script 2 for fluorescence/segmentation channels.
    """
    input:
        expand(
            f"{ANALYSIS}/3_improved_registration/dapi_blend/{{fish}}/c0_3d.tif",
            fish=FISH,
        ),
        expand(
            f"{ANALYSIS}/2_registered/{{fish}}/rigid_3d_c0.tif",
            fish=FISH,
        ),
    output:
        expand(
            f"{ANALYSIS}/4_registered/{{driving}}_{{stages}}_z10/{{fish}}/c0.tif",
            fish=FISH,
            allow_missing=True,
        )
    log:
        f"{ANALYSIS}/logs/script4_{{driving}}_{{stages}}.log"
    shell:
        "{PYTHON} 4_cross_fish_registration.py --steps register "
        "--driving {wildcards.driving} --stages {wildcards.stages} --z-spacing 10 "
        "> {log} 2>&1"


rule script4_evaluate:
    """
    Compute alignment metrics (NCC, Dice) across all script-4 experiments.
    Runs once after all registration jobs complete.
    """
    input:
        expand(
            f"{ANALYSIS}/4_registered/{{driving}}_{{stages}}_z10/{{fish}}/c0.tif",
            driving=SCRIPT4_DRIVING,
            stages=SCRIPT4_STAGES,
            fish=FISH,
        )
    output:
        f"{ANALYSIS}/4_registered/evaluation.csv",
        f"{ANALYSIS}/4_registered/evaluation_per_fish.csv",
    log:
        f"{ANALYSIS}/logs/script4_evaluate.log"
    shell:
        "{PYTHON} 4_cross_fish_registration.py --steps evaluate > {log} 2>&1"
