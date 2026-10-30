"""
conftest.py
-----------
Shared synthetic image fixtures for the test suite.

All test images are generated with NumPy so the tests have zero
dependency on external image datasets or sample files.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: create a solid-colour BGR image
# ---------------------------------------------------------------------------

def make_solid(height: int, width: int, bgr: tuple) -> np.ndarray:
    """Return a solid-colour BGR image of the given size."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = bgr
    return img


def make_noise(height: int, width: int, mean: int = 128, std: int = 40) -> np.ndarray:
    """Return a random-noise BGR image (looks like static)."""
    noise = np.random.normal(mean, std, (height, width, 3)).clip(0, 255).astype(np.uint8)
    return noise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_image() -> np.ndarray:
    """A realistic noise image with good lighting (mean ≈ 128)."""
    return make_noise(480, 640, mean=128, std=50)


@pytest.fixture
def dark_image() -> np.ndarray:
    """An almost completely dark image (mean ≈ 10)."""
    return make_solid(480, 640, bgr=(10, 10, 10))


@pytest.fixture
def very_dark_image() -> np.ndarray:
    """A fully black image."""
    return make_solid(480, 640, bgr=(0, 0, 0))


@pytest.fixture
def bright_glare_image() -> np.ndarray:
    """An almost completely white/overexposed image."""
    return make_solid(480, 640, bgr=(255, 255, 255))


@pytest.fixture
def blurry_image() -> np.ndarray:
    """A very smooth image (low Laplacian variance) – simulates blur."""
    # A uniform low-std image has very few edges → low Laplacian variance
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add tiny amount of noise to avoid completely degenerate arrays
    noise = np.random.randint(0, 3, img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sharp_image() -> np.ndarray:
    """A checkerboard pattern – very high Laplacian variance (sharp edges)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tile = 20
    for r in range(0, 480, tile):
        for c in range(0, 640, tile):
            colour = 255 if ((r // tile + c // tile) % 2 == 0) else 0
            img[r:r+tile, c:c+tile] = colour
    return img


@pytest.fixture
def low_res_image() -> np.ndarray:
    """A tiny 80×60 image – well below 640×480 minimum."""
    return make_noise(60, 80, mean=128, std=40)


@pytest.fixture
def high_res_image() -> np.ndarray:
    """A 1920×1080 noise image."""
    return make_noise(1080, 1920, mean=128, std=40)


@pytest.fixture
def partial_glare_image() -> np.ndarray:
    """An image where ~10% of pixels are bright white (glare hotspot)."""
    img = make_noise(480, 640, mean=100, std=30)
    # Inject a bright white patch covering ~10% of total pixels.
    # 480*640 = 307200 total; need >5% = 15360 pixels.
    # Use a patch of height=96 (~20%), width=160 (~25%) = 15360 pixels exactly.
    patch_h = int(480 * 0.20)
    patch_w = int(640 * 0.13)   # 96*83 ≈ 7968 glare px – let's be generous
    patch_h = 100
    patch_w = 200               # 100*200 = 20000 px > 15360 (5%) – definitely fails
    img[:patch_h, :patch_w] = 255
    return img


@pytest.fixture
def uniform_obstruction_image() -> np.ndarray:
    """
    A completely uniform grey image.
    Simulates a camera with a blocked lens or pointing at a uniform surface.
    """
    return make_solid(480, 640, bgr=(100, 100, 100))
