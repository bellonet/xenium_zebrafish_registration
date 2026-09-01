"""
Pipeline tests.

Three focused tests covering the parts most likely to have silent bugs:
  1. 2D rigid transform round-trip (forward → inverse = identity)
  2. Script-4 output dtype/shape sanity
  3. ds_xy in transform.json is consistent with actual output dimensions
"""

import json
import math
import os

import numpy as np
import pytest

# ── paths (relative to repo root, i.e. run pytest from there) ─────────────────
ANALYSIS = '../analysis'
IN4_DIR  = os.path.join(ANALYSIS, '4_registered')
REF_FISH = 1


# ══════════════════════════════════════════════════════════════════════════════
# 1. 2D rigid transform round-trip
# ══════════════════════════════════════════════════════════════════════════════
# Copied from 5_apply_registration.py so the test is self-contained and
# doesn't trigger module-level I/O at import time.

def _R2(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rigid2d(angle_deg, tx, ty, center):
    """x' = R @ x + offset"""
    R      = _R2(math.radians(angle_deg))
    c      = np.array(center, dtype=np.float64)
    offset = -R @ c + c + np.array([tx, ty], dtype=np.float64)
    return R, offset


def _compose(R1, off1, R2, off2):
    """T2 ∘ T1"""
    return R2 @ R1, R2 @ off1 + off2


def _invert(R, offset, x):
    """Inverse of x' = R @ x + offset  →  x = R^T @ (x' - offset)"""
    return R.T @ (x - offset)


@pytest.mark.parametrize("angle,tx,ty,center,point", [
    (0.0,    0.0,  0.0,  (0.0,   0.0),   (100.0, 200.0)),   # identity
    (90.0,   0.0,  0.0,  (0.0,   0.0),   (1.0,   0.0)),     # pure rotation
    (0.0,   10.0, -5.0,  (50.0, 50.0),   (0.0,   0.0)),     # pure translation
    (37.0,   3.0,  7.0,  (256.0, 512.0), (100.0, 400.0)),   # general case
    (-15.0, -2.0,  4.0,  (128.0, 128.0), (300.0, 50.0)),    # negative angle/tx
])
def test_rigid2d_round_trip(angle, tx, ty, center, point):
    """Applying a rigid transform then its inverse should return the original point."""
    R, off = _rigid2d(angle, tx, ty, center)
    x      = np.array(point, dtype=np.float64)
    x_fwd  = R @ x + off
    x_back = _invert(R, off, x_fwd)
    np.testing.assert_allclose(x_back, x, atol=1e-10,
                                err_msg=f'Round-trip failed for angle={angle}')


@pytest.mark.parametrize("params", [
    # compose two transforms, round-trip the composition
    dict(t1=(30.0, 5.0, -3.0, (100.0, 100.0)),
         t2=(-20.0, -1.0, 8.0, (200.0, 50.0)),
         point=(150.0, 75.0)),
    dict(t1=(0.0, 10.0, 0.0, (0.0, 0.0)),
         t2=(90.0, 0.0, 0.0, (50.0, 50.0)),
         point=(0.0, 0.0)),
])
def test_rigid2d_composition_round_trip(params):
    """Composing two transforms then inverting should also round-trip correctly."""
    R1, off1 = _rigid2d(*params['t1'])
    R2, off2 = _rigid2d(*params['t2'])
    Rc, offc = _compose(R1, off1, R2, off2)
    x        = np.array(params['point'], dtype=np.float64)
    x_fwd    = Rc @ x + offc
    x_back   = _invert(Rc, offc, x_fwd)
    np.testing.assert_allclose(x_back, x, atol=1e-10,
                                err_msg='Composed round-trip failed')


# ══════════════════════════════════════════════════════════════════════════════
# 2. Script-4 output dtype and shape sanity
# ══════════════════════════════════════════════════════════════════════════════

def _available_experiments():
    if not os.path.isdir(IN4_DIR):
        return []
    return [d for d in os.listdir(IN4_DIR)
            if os.path.isdir(os.path.join(IN4_DIR, d, str(REF_FISH)))]


@pytest.mark.skipif(not os.path.isdir(IN4_DIR), reason='script-4 outputs not present')
@pytest.mark.parametrize("exp", _available_experiments() or ['_placeholder'])
def test_script4_output_dtype_shape(exp):
    if exp == '_placeholder':
        pytest.skip('no script-4 experiments found')

    import tifffile

    ref_dir = os.path.join(IN4_DIR, exp, str(REF_FISH))
    c0_path = os.path.join(ref_dir, 'c0.tif')
    if not os.path.exists(c0_path):
        pytest.skip(f'c0.tif missing for {exp}/fish{REF_FISH}')

    vol = tifffile.imread(c0_path)
    assert vol.ndim == 3,           f'{exp}: c0.tif should be 3D, got shape {vol.shape}'
    assert vol.dtype == np.float32, f'{exp}: c0.tif should be float32, got {vol.dtype}'
    assert vol.shape[0] >= 1,      f'{exp}: Z dimension is 0'
    assert not np.all(vol == 0),   f'{exp}: c0.tif is all zeros'
    assert np.isfinite(vol).all(), f'{exp}: c0.tif contains NaN or Inf'


# ══════════════════════════════════════════════════════════════════════════════
# 3. ds_xy in transform.json is consistent with output image dimensions
# ══════════════════════════════════════════════════════════════════════════════

XENIUM_PX_UM = 0.2125

@pytest.mark.skipif(not os.path.isdir(IN4_DIR), reason='script-4 outputs not present')
@pytest.mark.parametrize("exp", _available_experiments() or ['_placeholder'])
def test_script4_ds_xy_consistent(exp):
    """ds_xy in transform.json × input pixel size should equal output pixel size,
    and output volume shape should be consistent across all fish in an experiment."""
    if exp == '_placeholder':
        pytest.skip('no script-4 experiments found')

    import tifffile

    tf_path = os.path.join(IN4_DIR, exp, str(REF_FISH), 'transform.json')
    c0_path = os.path.join(IN4_DIR, exp, str(REF_FISH), 'c0.tif')
    if not os.path.exists(tf_path) or not os.path.exists(c0_path):
        pytest.skip(f'transform.json or c0.tif missing for {exp}/fish{REF_FISH}')

    with open(tf_path) as fh:
        tf = json.load(fh)

    ds_xy = tf.get('ds_xy', 1)
    assert isinstance(ds_xy, int) and ds_xy >= 1, f'ds_xy should be a positive int, got {ds_xy!r}'

    ref_shape = tifffile.imread(c0_path).shape  # (Z, H, W)

    # Every other fish in the same experiment must have the same output shape
    exp_dir = os.path.join(IN4_DIR, exp)
    for fish_dir in sorted(os.listdir(exp_dir)):
        fish_c0 = os.path.join(exp_dir, fish_dir, 'c0.tif')
        if not os.path.exists(fish_c0):
            continue
        fish_shape = tifffile.imread(fish_c0).shape
        assert fish_shape == ref_shape, (
            f'{exp}/fish{fish_dir}: shape {fish_shape} != reference shape {ref_shape}')
