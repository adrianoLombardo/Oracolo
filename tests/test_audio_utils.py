import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OcchioOnniveggente.src.hardware.audio import apply_agc, apply_limiter


def test_apply_agc_and_limiter():
    x = np.linspace(-0.01, 0.01, 1000, dtype=np.float32)
    y = apply_agc(x, target_rms=0.1)
    assert y.dtype == np.float32
    assert np.sqrt(np.mean(y**2)) > np.sqrt(np.mean(x**2))
    assert np.max(np.abs(y)) <= 1.0

    x2 = np.array([0.0, 2.0, -2.0], dtype=np.float32)
    y2 = apply_limiter(x2, threshold=0.5)
    assert np.max(np.abs(y2)) <= 0.5 + 1e-6
