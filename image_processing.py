"""
Image loading, pseudocolor LUT rendering, scalebar overlay, and dual export pipelines
for legacy microscope batch processing.

Pipelines:
1. Visual Presentation Pipeline:
   - Normalizes each grayscale channel array according to bit depth / display window & tonal settings
   - Applies pseudocolor LUT based on channel keyword:
     * 'blue'         -> DAPI style (Blue LUT)
     * 'green'        -> FITC style (Green LUT)
     * 'red'          -> TRITC style (Red LUT)
     * 'bright field' -> Grayscale / Phase contrast
   - Blends selected channels into an 8-bit RGB composite
   - Burns in the calibrated scalebar overlay (Objective Lens, Length in μm, Calibration Factor in μm/pixel)
   - Exports as PNG, JPEG, or 8-bit RGB TIFF

2. Raw Analysis Pipeline ('Save for Analysis'):
   - Uses `tifffile` to export selected channels as a raw, multi-dimensional TIFF (`(C, Y, X)` or `(Y, X)`).
   - Strictly preserves original grayscale intensity values and native integer dtype (`uint8` / `uint16`).
   - Zero LUTs, zero tonal adjustments, zero scalebar, zero channel flattening.
"""

from dataclasses import dataclass
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

from batch_grouping import ALLOWED_CHANNELS


# Objective lens options required by specification
OBJECTIVE_LENSES: Tuple[str, ...] = ("4x", "10x", "16x", "20x", "40x", "100x")

# Pseudocolor LUT definitions for each allowed channel keyword
@dataclass(frozen=True)
class PseudocolorLUT:
    channel_key: str
    style_name: str
    rgb_weights: Tuple[float, float, float]
    hex_color: str
    description: str


CHANNEL_LUTS: Dict[str, PseudocolorLUT] = {
    "blue": PseudocolorLUT(
        channel_key="blue",
        style_name="DAPI Style (Blue)",
        rgb_weights=(0.0, 0.0, 1.0),
        hex_color="#0000FF",
        description="Nuclear counterstain LUT (DAPI / Hoechst — λ_em ≈ 461 nm)",
    ),
    "green": PseudocolorLUT(
        channel_key="green",
        style_name="FITC Style (Green)",
        rgb_weights=(0.0, 1.0, 0.0),
        hex_color="#00FF00",
        description="Green fluorescence LUT (FITC / GFP / Alexa 488 — λ_em ≈ 519 nm)",
    ),
    "red": PseudocolorLUT(
        channel_key="red",
        style_name="TRITC Style (Red)",
        rgb_weights=(1.0, 0.0, 0.0),
        hex_color="#FF0000",
        description="Red fluorescence LUT (TRITC / Texas Red / mCherry — λ_em ≈ 576 nm)",
    ),
    "bright field": PseudocolorLUT(
        channel_key="bright field",
        style_name="Grayscale / Phase Contrast",
        rgb_weights=(1.0, 1.0, 1.0),
        hex_color="#FFFFFF",
        description="Transmitted light monochrome LUT (Bright Field / Phase Contrast / DIC)",
    ),
}


@dataclass
class ChannelRenderConfig:
    """Per-channel visual adjustment parameters."""
    brightness: float = 0.0     # [-100.0, +100.0]
    contrast: float = 0.0       # [-100.0, +100.0]
    gamma: float = 1.0          # [0.20, 3.00]
    auto_contrast: bool = False
    weight: float = 1.0         # [0.0, 1.0] channel blend intensity


@dataclass
class ScalebarConfig:
    """Scalebar overlay settings."""
    enabled: bool = True
    objective_lens: str = "40x"
    length_um: float = 50.0
    calibration_um_per_px: float = 0.1625
    position: str = "Bottom Right"
    color: str = "White"
    show_backing_box: bool = True
    include_objective_in_label: bool = True


def load_raw_grayscale_array(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Load a native grayscale TIFF or JPEG file from bytes into an immutable 2D numpy array.

    Strictly preserves the native pixel dtype (`uint8`, `uint16`, etc.) and raw intensity values.
    If a file was saved with redundant RGB channels (e.g., an 8-bit grayscale JPEG saved as 3-channel RGB),
    extracts the 2D grayscale plane while keeping its native integer dtype.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buf = io.BytesIO(file_bytes)

    arr: Optional[np.ndarray] = None
    if ext in ("tif", "tiff"):
        try:
            arr = tifffile.imread(buf)
        except Exception:
            buf.seek(0)
            pil_img = Image.open(buf)
            arr = np.asarray(pil_img)
    else:
        pil_img = Image.open(buf)
        arr = np.asarray(pil_img)

    if arr is None:
        raise ValueError(f"Could not decode image data from '{filename}'.")

    orig_dtype = arr.dtype

    # Ensure a 2D grayscale array (H, W) while preserving original integer dtype
    if arr.ndim == 2:
        raw_2d = np.array(arr, dtype=orig_dtype, copy=True)
    elif arr.ndim == 3:
        # Case 1: (H, W, C) where C in (3, 4) — e.g. grayscale JPEG/TIFF stored as RGB/RGBA
        if arr.shape[2] in (3, 4) and arr.shape[0] > 4 and arr.shape[1] > 4:
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                raw_2d = np.array(arr[..., 0], dtype=orig_dtype, copy=True)
            else:
                # Luminance conversion preserving original integer dtype
                lum = (
                    0.299 * arr[..., 0].astype(np.float64)
                    + 0.587 * arr[..., 1].astype(np.float64)
                    + 0.114 * arr[..., 2].astype(np.float64)
                )
                if np.issubdtype(orig_dtype, np.integer):
                    info = np.iinfo(orig_dtype)
                    raw_2d = np.clip(np.round(lum), info.min, info.max).astype(orig_dtype)
                else:
                    raw_2d = lum.astype(orig_dtype)
        # Case 2: (1, H, W) single-slice 3D TIFF
        elif arr.shape[0] == 1:
            raw_2d = np.array(arr[0], dtype=orig_dtype, copy=True)
        # Case 3: Multi-slice Z-stack (Z, H, W) -> maximum intensity projection preserving dtype
        else:
            raw_2d = np.max(arr, axis=0).astype(orig_dtype, copy=True)
    else:
        # Squeeze higher-dimensional singleton axes if present
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            raw_2d = np.array(squeezed, dtype=orig_dtype, copy=True)
        else:
            raise ValueError(
                f"Unsupported image dimensions {arr.shape} in '{filename}'. Expected 2D grayscale image."
            )

    raw_2d.flags.writeable = False
    return raw_2d


def normalize_and_adjust_channel(
    raw_2d: np.ndarray,
    config: Optional[ChannelRenderConfig] = None,
) -> np.ndarray:
    """
    Convert a raw 2D grayscale array into a normalized float32 plane in [0.0, 1.0]
    with optional Brightness, Contrast, Gamma, and Auto-Contrast adjustments.
    Does NOT modify `raw_2d`.
    """
    if config is None:
        config = ChannelRenderConfig()

    plane = raw_2d.astype(np.float32, copy=True)

    if config.auto_contrast and plane.size > 0:
        low, high = np.percentile(plane, [0.5, 99.5])
        if high <= low:
            low, high = float(np.min(plane)), float(np.max(plane))
        denom = max(float(high - low), 1e-6)
        norm = np.clip((plane - float(low)) / denom, 0.0, 1.0)
    else:
        # Normalize according to native bit-depth range or actual max value
        if raw_2d.dtype == np.uint8:
            norm = plane / 255.0
        elif raw_2d.dtype == np.uint16:
            max_val = float(np.max(plane)) if plane.size > 0 else 65535.0
            # Handle 12-bit (0..4095) or 14-bit (0..16383) data stored in 16-bit TIFF containers
            if 0 < max_val <= 4095.0:
                norm = plane / 4095.0
            elif 4095.0 < max_val <= 16383.0:
                norm = plane / 16383.0
            else:
                norm = plane / 65535.0
        else:
            p_min = float(np.min(plane)) if plane.size > 0 else 0.0
            p_max = float(np.max(plane)) if plane.size > 0 else 1.0
            denom = max(p_max - p_min, 1e-6)
            norm = (plane - p_min) / denom

        norm = np.clip(norm, 0.0, 1.0)

    # Apply Brightness [-100, +100] and Contrast [-100, +100]
    if abs(config.contrast) > 1e-4 or abs(config.brightness) > 1e-4:
        c_norm = float(np.clip(config.contrast / 100.0, -0.95, 0.95))
        contrast_factor = float(np.tan((c_norm + 1.0) * (np.pi / 4.0)))
        b_offset = float(config.brightness / 100.0) * 0.5
        norm = contrast_factor * (norm - 0.5 + b_offset) + 0.5
        norm = np.clip(norm, 0.0, 1.0)

    # Apply Gamma [0.20, 3.00]
    gamma_val = max(float(config.gamma), 0.1)
    if abs(gamma_val - 1.0) > 1e-4:
        norm = np.power(norm, 1.0 / gamma_val)

    # Apply channel weight
    weight = float(np.clip(config.weight, 0.0, 1.0))
    return np.clip(norm * weight, 0.0, 1.0)


def apply_channel_lut(
    normalized_plane: np.ndarray,
    channel_key: str,
) -> np.ndarray:
    """
    Apply the pseudocolor LUT for `channel_key` ('blue', 'green', 'red', 'bright field')
    to a normalized 2D float32 plane in [0.0, 1.0], returning a float32 RGB array (H, W, 3) in [0.0, 1.0].
    """
    lut = CHANNEL_LUTS.get(channel_key.lower(), CHANNEL_LUTS["bright field"])
    r_w, g_w, b_w = lut.rgb_weights
    rgb = np.empty((*normalized_plane.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized_plane * r_w
    rgb[..., 1] = normalized_plane * g_w
    rgb[..., 2] = normalized_plane * b_w
    return rgb


def render_single_channel_preview(
    raw_2d: np.ndarray,
    channel_key: str,
    config: Optional[ChannelRenderConfig] = None,
) -> np.ndarray:
    """Render a single channel with its pseudocolor LUT to an 8-bit RGB image (H, W, 3)."""
    norm = normalize_and_adjust_channel(raw_2d, config)
    rgb_float = apply_channel_lut(norm, channel_key)
    return (np.clip(rgb_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def render_group_composite(
    raw_channels: Dict[str, np.ndarray],
    selected_channels: Sequence[str],
    channel_configs: Optional[Dict[str, ChannelRenderConfig]] = None,
) -> np.ndarray:
    """
    Render and merge the selected channels for a group into an 8-bit RGB composite image (H, W, 3).

    Uses additive fluorescence superposition over the active channels.
    If a group's channels differ in dimensions, resizes secondary channels to match the first channel's shape
    for display rendering (raw analysis export never resizes).
    """
    active = [ch for ch in ALLOWED_CHANNELS if ch in selected_channels and ch in raw_channels]
    if not active:
        # Fallback if no channel selected: return black canvas matching first available channel
        if raw_channels:
            first_arr = next(iter(raw_channels.values()))
            return np.zeros((first_arr.shape[0], first_arr.shape[1], 3), dtype=np.uint8)
        return np.zeros((256, 256, 3), dtype=np.uint8)

    ref_shape = raw_channels[active[0]].shape
    h, w = ref_shape[0], ref_shape[1]
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for ch in active:
        raw_arr = raw_channels[ch]
        cfg = channel_configs.get(ch) if channel_configs else None
        norm = normalize_and_adjust_channel(raw_arr, cfg)

        if norm.shape != (h, w):
            pil_resized = Image.fromarray((norm * 255.0).astype(np.uint8)).resize(
                (w, h), resample=Image.Resampling.BILINEAR
            )
            norm = np.asarray(pil_resized, dtype=np.float32) / 255.0

        rgb_plane = apply_channel_lut(norm, ch)
        composite += rgb_plane

    composite = np.clip(composite, 0.0, 1.0)
    return (composite * 255.0 + 0.5).astype(np.uint8)


def draw_scalebar_overlay(
    rgb_uint8: np.ndarray,
    scalebar_cfg: ScalebarConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Burn a calibrated scalebar into a copy of `rgb_uint8`.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (rendered_rgb_uint8, info_dict) where info_dict reports pixel length, FOV in μm,
        and whether the requested scalebar exceeded the image width.
    """
    h, w, _ = rgb_uint8.shape
    cal = float(scalebar_cfg.calibration_um_per_px)
    length_um = float(scalebar_cfg.length_um)

    if cal <= 0 or length_um <= 0:
        return rgb_uint8.copy(), {
            "drawn": False,
            "bar_px": 0,
            "fov_width_um": 0.0,
            "clamped": False,
        }

    fov_width_um = w * cal
    fov_height_um = h * cal
    raw_bar_px = int(round(length_um / cal))

    margin_x = max(16, int(w * 0.04))
    margin_y = max(16, int(h * 0.04))
    max_allowed_px = max(10, w - 2 * margin_x)

    clamped = raw_bar_px > max_allowed_px
    bar_px = int(np.clip(raw_bar_px, 4, max_allowed_px))

    if not scalebar_cfg.enabled:
        return rgb_uint8.copy(), {
            "drawn": False,
            "bar_px": raw_bar_px,
            "fov_width_um": fov_width_um,
            "fov_height_um": fov_height_um,
            "clamped": clamped,
        }

    pil_img = Image.fromarray(rgb_uint8.copy())
    draw = ImageDraw.Draw(pil_img)

    bar_height = max(5, int(h * 0.014))
    font_size = max(13, int(min(h, w) * 0.032))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Format label text
    len_str = f"{int(length_um)}" if length_um.is_integer() else f"{length_um:g}"
    if scalebar_cfg.include_objective_in_label and scalebar_cfg.objective_lens:
        label_text = f"{len_str} μm ({scalebar_cfg.objective_lens})"
    else:
        label_text = f"{len_str} μm"

    # Measure text bounding box
    try:
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(label_text) * (font_size // 2)
        text_h = font_size

    pos = scalebar_cfg.position
    if "Left" in pos:
        x1 = margin_x
        x2 = margin_x + bar_px
    else:
        x2 = w - margin_x
        x1 = x2 - bar_px

    if "Top" in pos:
        y1 = margin_y + text_h + 8
        y2 = y1 + bar_height
    else:
        y2 = h - margin_y
        y1 = y2 - bar_height

    text_x = x1 + (bar_px - text_w) / 2.0
    text_x = max(6.0, min(float(w - text_w - 6), text_x))
    text_y = y1 - text_h - 6

    color_map = {
        "White": (255, 255, 255),
        "Yellow": (255, 235, 59),
        "Cyan": (0, 229, 255),
        "Green": (0, 255, 100),
        "Black": (0, 0, 0),
    }
    fg_color = color_map.get(scalebar_cfg.color, (255, 255, 255))
    bg_color = (240, 240, 240) if scalebar_cfg.color == "Black" else (0, 0, 0)

    if scalebar_cfg.show_backing_box:
        pad_x = 8
        pad_y = 6
        box_x1 = min(x1, int(text_x)) - pad_x
        box_x2 = max(x2, int(text_x + text_w)) + pad_x
        box_y1 = int(text_y) - pad_y
        box_y2 = y2 + pad_y
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=bg_color)

    # Draw scalebar rectangle and label
    draw.rectangle([x1, y1, x2, y2], fill=fg_color)
    draw.text((text_x, text_y), label_text, fill=fg_color, font=font)

    return np.asarray(pil_img, dtype=np.uint8), {
        "drawn": True,
        "bar_px": raw_bar_px,
        "effective_bar_px": bar_px,
        "fov_width_um": fov_width_um,
        "fov_height_um": fov_height_um,
        "clamped": clamped,
        "label_text": label_text,
    }


# ==============================================================================
# EXPORT PIPELINE 1: SAVE FOR ANALYSIS (RAW MULTI-DIMENSIONAL TIFF VIA TIFFFILE)
# ==============================================================================

def export_group_for_analysis(
    raw_channels: Dict[str, np.ndarray],
    selected_channels: Sequence[str],
    calibration_um_per_px: Optional[float] = None,
    objective_lens: Optional[str] = None,
) -> bytes:
    """
    Export the selected channels for a group as a raw, multi-dimensional TIFF using `tifffile`.

    Guarantees:
    - Strictly preserves original grayscale intensity data (`uint8` or `uint16`).
    - No LUTs, no pseudocoloring, no brightness/contrast/gamma tweaks, no scalebar, no channel flattening.
    - Stores multi-channel data as `(C, Y, X)` (`axes='CYX'`) with channel labels in TIFF metadata.
    """
    active = [ch for ch in ALLOWED_CHANNELS if ch in selected_channels and ch in raw_channels]
    if not active:
        raise ValueError("Select at least one channel to export for analysis.")

    arrays = [raw_channels[ch] for ch in active]

    # Promote to the widest native integer dtype among the selected channels (e.g. uint16 if mixed uint8+uint16)
    target_dtype = np.result_type(*[a.dtype for a in arrays])
    if not np.issubdtype(target_dtype, np.integer):
        target_dtype = np.uint16

    typed_arrays = [np.asarray(a, dtype=target_dtype) for a in arrays]

    # Check if all selected channels share the same (H, W) shape
    first_shape = typed_arrays[0].shape
    same_shape = all(a.shape == first_shape for a in typed_arrays)

    buf = io.BytesIO()
    if same_shape:
        if len(typed_arrays) == 1:
            # Single channel 2D grayscale array (Y, X)
            volume = typed_arrays[0]
            axes = "YX"
        else:
            # Multi-dimensional stack of shape (C, Y, X) — zero flattening!
            volume = np.stack(typed_arrays, axis=0)
            axes = "CYX"

        metadata: Dict[str, Any] = {
            "axes": axes,
            "Labels": list(active),
        }
        if objective_lens:
            metadata["Objective"] = str(objective_lens)

        resolution = None
        if calibration_um_per_px and calibration_um_per_px > 0:
            px_per_um = 1.0 / float(calibration_um_per_px)
            resolution = (px_per_um, px_per_um)
            metadata["unit"] = "um"

        if resolution is not None:
            tifffile.imwrite(
                buf,
                volume,
                imagej=True,
                photometric="minisblack",
                resolution=resolution,
                metadata=metadata,
            )
        else:
            tifffile.imwrite(
                buf,
                volume,
                imagej=True,
                photometric="minisblack",
                metadata=metadata,
            )
    else:
        # Fallback if channels have different spatial resolutions: write each raw plane as a separate page
        with tifffile.TiffWriter(buf) as tif:
            for ch, arr in zip(active, typed_arrays):
                tif.write(
                    arr,
                    photometric="minisblack",
                    description=f"Channel={ch};RawAnalysis=True",
                )

    return buf.getvalue()


# ==============================================================================
# EXPORT PIPELINE 2: SAVE FOR PRESENTATION (PSEUDOCOLORED + MERGED + SCALEBAR)
# ==============================================================================

def export_group_for_presentation(
    raw_channels: Dict[str, np.ndarray],
    selected_channels: Sequence[str],
    output_format: str = "PNG",
    channel_configs: Optional[Dict[str, ChannelRenderConfig]] = None,
    scalebar_cfg: Optional[ScalebarConfig] = None,
) -> Tuple[bytes, str, str]:
    """
    Export the visually pseudocolored and merged composite image with the scalebar burned in.

    Parameters
    ----------
    raw_channels : Dict[str, np.ndarray]
        Dictionary mapping channel keyword to raw 2D grayscale array.
    selected_channels : Sequence[str]
        Channels chosen by the user to include in the merged composite.
    output_format : str
        One of 'PNG', 'JPEG', or '8-bit RGB TIFF'.
    channel_configs : Optional[Dict[str, ChannelRenderConfig]]
        Per-channel brightness/contrast/gamma adjustments.
    scalebar_cfg : Optional[ScalebarConfig]
        Scalebar configuration to burn into the image.

    Returns
    -------
    Tuple[bytes, str, str]
        (encoded_bytes, mime_type, file_extension)
    """
    active = [ch for ch in ALLOWED_CHANNELS if ch in selected_channels and ch in raw_channels]
    if not active:
        raise ValueError("Select at least one channel to export for presentation.")

    merged_rgb = render_group_composite(
        raw_channels=raw_channels,
        selected_channels=active,
        channel_configs=channel_configs,
    )

    if scalebar_cfg is not None and scalebar_cfg.enabled:
        merged_rgb, _ = draw_scalebar_overlay(merged_rgb, scalebar_cfg)

    render_arr = np.asarray(merged_rgb, dtype=np.uint8)
    fmt = output_format.strip().upper()
    buf = io.BytesIO()

    if fmt in ("JPEG", "JPG"):
        pil_img = Image.fromarray(render_arr)
        pil_img.save(buf, format="JPEG", quality=95, subsampling=0)
        return buf.getvalue(), "image/jpeg", "jpg"
    elif "TIFF" in fmt or "TIF" in fmt:
        tifffile.imwrite(buf, render_arr, photometric="rgb")
        return buf.getvalue(), "image/tiff", "tif"
    else:
        pil_img = Image.fromarray(render_arr)
        pil_img.save(buf, format="PNG")
        return buf.getvalue(), "image/png", "png"
