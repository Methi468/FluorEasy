"""
Synthetic legacy microscope batch generator for testing and interactive demonstration.

Generates realistic multi-channel grayscale TIFF (16-bit uint16) and JPEG (8-bit uint8)
files following the legacy naming convention:
- `<Prefix>_bright field.tif`
- `<Prefix>_blue.tif`
- `<Prefix>_green.tif`
- `<Prefix>_red.tif`
"""

import io
import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import tifffile


def _make_cell_field(
    height: int = 420,
    width: int = 560,
    seed: int = 42,
    num_cells: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize 4 realistic registered microscopy planes in float32 [0.0, 1.0]:
    (bright_field, blue_dapi, green_fitc, red_tritc)
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    dapi = np.zeros((height, width), dtype=np.float32)
    fitc = np.zeros((height, width), dtype=np.float32)
    tritc = np.zeros((height, width), dtype=np.float32)
    phase = np.full((height, width), 0.22, dtype=np.float32)

    margin = 55
    centers_y = rng.uniform(margin, height - margin, size=num_cells)
    centers_x = rng.uniform(margin, width - margin, size=num_cells)

    for cy, cx in zip(centers_y, centers_x):
        # Elliptical rotation
        theta = rng.uniform(0, np.pi)
        dy = yy - cy
        dx = xx - cx
        ry = dy * np.cos(theta) - dx * np.sin(theta)
        rx = dy * np.sin(theta) + dx * np.cos(theta)

        # 1. DAPI nucleus (Blue): crisp ovoid with nucleolar chromatin texture
        nuc_a = rng.uniform(16.0, 24.0)
        nuc_b = rng.uniform(20.0, 30.0)
        r2_nuc = (ry / nuc_a) ** 2 + (rx / nuc_b) ** 2
        nucleus = np.exp(-0.5 * (r2_nuc ** 1.6)) * rng.uniform(0.7, 0.98)
        dapi = np.maximum(dapi, nucleus)

        # 2. FITC cytoplasm / cytoskeleton (Green): larger cell body + radial filaments
        cyto_a = nuc_a * rng.uniform(1.9, 2.5)
        cyto_b = nuc_b * rng.uniform(2.0, 2.7)
        r2_cyto = (ry / cyto_a) ** 2 + (rx / cyto_b) ** 2
        angle = np.arctan2(ry, rx)
        filaments = 0.65 + 0.35 * np.cos(angle * rng.integers(6, 12)) ** 2
        cyto_shell = np.exp(-0.5 * r2_cyto) * (1.0 - 0.55 * np.exp(-0.5 * r2_nuc)) * filaments
        fitc = np.maximum(fitc, cyto_shell * rng.uniform(0.65, 0.95))

        # 3. TRITC mitochondria / vesicles (Red): perinuclear puncta
        num_puncta = rng.integers(10, 20)
        for _ in range(num_puncta):
            ang = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.85, 1.85)
            py = cy + np.sin(ang) * nuc_a * dist
            px = cx + np.cos(ang) * nuc_b * dist
            sigma = rng.uniform(2.2, 4.2)
            spot = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2.0 * sigma ** 2))
            tritc = np.maximum(tritc, spot * rng.uniform(0.55, 0.98))

        # 4. Bright field / Phase contrast: bright phase halo around cell boundary + dark organelle core
        halo_ring = np.exp(-((np.sqrt(r2_cyto) - 0.95) ** 2) / 0.04) * 0.25
        cell_shade = -np.exp(-0.5 * r2_cyto) * 0.08 + np.exp(-0.5 * r2_nuc) * 0.06
        phase = phase + halo_ring + cell_shade

    # Add subtle camera sensor readout noise
    dapi = np.clip(dapi + rng.normal(0.015, 0.008, size=(height, width)), 0.0, 1.0)
    fitc = np.clip(fitc + rng.normal(0.015, 0.008, size=(height, width)), 0.0, 1.0)
    tritc = np.clip(tritc + rng.normal(0.012, 0.007, size=(height, width)), 0.0, 1.0)
    phase = np.clip(phase + rng.normal(0.0, 0.012, size=(height, width)), 0.0, 1.0)

    return phase.astype(np.float32), dapi.astype(np.float32), fitc.astype(np.float32), tritc.astype(np.float32)


def _encode_16bit_tiff(norm_plane: np.ndarray) -> bytes:
    arr16 = np.clip(norm_plane * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr16, photometric="minisblack")
    return buf.getvalue()


def _encode_8bit_jpeg(norm_plane: np.ndarray) -> bytes:
    arr8 = np.clip(norm_plane * 255.0 + 0.5, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr8).save(buf, format="JPEG", quality=96)
    return buf.getvalue()


def generate_sample_batch_files() -> List[Tuple[str, bytes]]:
    """
    Generate a sample batch of 9 legacy microscope files across 3 groups:
    1. 'HeLa_Control_FOV1' (4 channels, 16-bit TIFF: bright field, blue, green, red)
    2. 'Neuron_Synapse_FOV2' (3 channels, 16-bit TIFF: blue, green, red)
    3. 'Fibroblast_Mitosis_FOV3' (2 channels, 8-bit JPEG: bright field, blue)
    """
    batch: List[Tuple[str, bytes]] = []

    # Group 1: 4 channels (16-bit TIFF)
    bf1, b1, g1, r1 = _make_cell_field(seed=101, num_cells=10)
    batch.append(("HeLa_Control_FOV1_bright field.tif", _encode_16bit_tiff(bf1)))
    batch.append(("HeLa_Control_FOV1_blue.tif", _encode_16bit_tiff(b1)))
    batch.append(("HeLa_Control_FOV1_green.tif", _encode_16bit_tiff(g1)))
    batch.append(("HeLa_Control_FOV1_red.tif", _encode_16bit_tiff(r1)))

    # Group 2: 3 channels (16-bit TIFF)
    _, b2, g2, r2 = _make_cell_field(seed=202, num_cells=8)
    batch.append(("Neuron_Synapse_FOV2_blue.tif", _encode_16bit_tiff(b2)))
    batch.append(("Neuron_Synapse_FOV2_green.tif", _encode_16bit_tiff(g2)))
    batch.append(("Neuron_Synapse_FOV2_red.tif", _encode_16bit_tiff(r2)))

    # Group 3: 2 channels (8-bit JPEG)
    bf3, b3, _, _ = _make_cell_field(seed=303, num_cells=7)
    batch.append(("Fibroblast_Mitosis_FOV3_bright field.jpg", _encode_8bit_jpeg(bf3)))
    batch.append(("Fibroblast_Mitosis_FOV3_blue.jpg", _encode_8bit_jpeg(b3)))

    return batch


def write_sample_files_to_disk(base_dir: str) -> str:
    """Write sample batch files to `<base_dir>/sample_images/` for convenient file-uploader testing."""
    out_dir = os.path.join(base_dir, "sample_images")
    os.makedirs(out_dir, exist_ok=True)
    for fname, raw_bytes in generate_sample_batch_files():
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "wb") as f:
                f.write(raw_bytes)
    return out_dir
