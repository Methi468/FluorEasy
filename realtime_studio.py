"""
Real-time HTML5 Canvas Studio for FluorEasy Batch Microscope Processor.

Provides the dark-themed, zero-latency interactive studio interface (matching the LifEasy
studio layout) tailored specifically for grouped legacy grayscale TIFF/JPEG channels:
- Assigned pseudocolor LUTs by filename channel keyword:
  * 'blue'         -> DAPI Style (Blue)
  * 'green'        -> FITC Style (Green)
  * 'red'          -> TRITC Style (Red)
  * 'bright field' -> Grayscale / Phase Contrast
- Real-time 60 FPS multi-channel additive blending, histogram, and individual channel strip
- Scalebar Panel with Objective Lens dropdown (4x, 10x, 16x, 20x, 40x, 100x),
  Scalebar Length (μm), and Calibration Factor (μm/pixel) with hardware-specific explanation
- Per-group channel selection checkboxes for:
  * 1A. Save for Analysis (Raw multi-dimensional TIFF via tifffile templates, preserving exact raw data)
  * 1B. Save for Presentation (Pseudocolored & merged with scalebar burned in: PNG, JPEG, or 8-bit RGB TIFF)
- Full Undo (Ctrl+Z / Cmd+Z) and Redo (Ctrl+Y / Cmd+Shift+Z) history stack + prominent Undo buttons
"""

import base64
import io
import json
from typing import Any, Dict, List, Optional
import numpy as np
import tifffile

from batch_grouping import ALLOWED_CHANNELS, ImageGroup
from image_processing import OBJECTIVE_LENSES, ScalebarConfig


STUDIO_LUT_PRESETS: Dict[str, Dict[str, Any]] = {
    "Alexa 405 / DAPI (Blue)": {
        "hex": "#3B82F6",
        "ex": 358,
        "em": 461,
        "desc": "UV/Violet excitation, blue emission (DAPI, Hoechst, Alexa Fluor 405)",
    },
    "Alexa 488 / GFP (Green)": {
        "hex": "#00FF44",
        "ex": 490,
        "em": 525,
        "desc": "Cyan-blue excitation, bright green emission (Alexa Fluor 488, FITC, EGFP)",
    },
    "Alexa 532 (Yellow-Green)": {
        "hex": "#CCFF00",
        "ex": 532,
        "em": 554,
        "desc": "Green-yellow emission fluorophore (Alexa Fluor 532)",
    },
    "Alexa 546 / Cy3 (Yellow-Orange)": {
        "hex": "#FFA500",
        "ex": 556,
        "em": 573,
        "desc": "Orange emission dye (Alexa Fluor 546, TRITC, Cy3)",
    },
    "Alexa 568 (Orange-Red)": {
        "hex": "#FF4500",
        "ex": 578,
        "em": 603,
        "desc": "Yellow-orange excitation, orange-red emission at 603 nm (Alexa Fluor 568)",
    },
    "Alexa 594 / mCherry (Red)": {
        "hex": "#FF1A1A",
        "ex": 590,
        "em": 617,
        "desc": "Orange-red excitation, deep red emission at 617 nm (Alexa Fluor 594, Texas Red)",
    },
    "Alexa 647 (Magenta)": {
        "hex": "#FF00FF",
        "ex": 650,
        "em": 665,
        "desc": "Far-red emission at 665 nm mapped to high-visibility Magenta (Alexa Fluor 647, Cy5)",
    },
    "Alexa 647 (Far-Red)": {
        "hex": "#E60033",
        "ex": 650,
        "em": 665,
        "desc": "Far-red emission at 665 nm mapped to spectral Crimson/Far-Red (Alexa Fluor 647)",
    },
    "Alexa 680 (Deep Crimson)": {
        "hex": "#B30047",
        "ex": 679,
        "em": 702,
        "desc": "Near-IR emission at 702 nm (Alexa Fluor 680)",
    },
    "Cyan (CFP / Aqua)": {
        "hex": "#00FFFF",
        "ex": 435,
        "em": 477,
        "desc": "Colorblind-safe Cyan LUT",
    },
    "Yellow (YFP / Gold)": {
        "hex": "#FFFF00",
        "ex": 514,
        "em": 527,
        "desc": "Bright Yellow LUT (EYFP, Venus)",
    },
    "Grayscale / White": {
        "hex": "#FFFFFF",
        "ex": None,
        "em": None,
        "desc": "Neutral monochrome display (DIC / Brightfield / Phase Contrast)",
    },
    "Custom Color": {
        "hex": "#00E5FF",
        "ex": None,
        "em": None,
        "desc": "User-defined RGB hex pseudocolor",
    },
}

DEFAULT_PRESET_FOR_KEYWORD: Dict[str, str] = {
    "blue": "Alexa 405 / DAPI (Blue)",
    "green": "Alexa 488 / GFP (Green)",
    "red": "Alexa 594 / mCherry (Red)",
    "bright field": "Grayscale / White",
}


def _build_tifffile_imagej_templates(
    num_channels: int,
    height: int,
    width: int,
    calibration_um_per_px: float,
) -> Dict[str, Dict[str, str]]:
    """
    Use `tifffile.imwrite(..., imagej=True)` to pre-generate exact ImageJ hyperstack
    binary prefix/suffix templates for 1..num_channels and bit depth (8 or 16).
    """
    templates: Dict[str, Dict[str, str]] = {}
    resolution = None
    if calibration_um_per_px and calibration_um_per_px > 0:
        px_per_um = 1.0 / float(calibration_um_per_px)
        resolution = (px_per_um, px_per_um)

    for bits, dt in ((8, np.uint8), (16, np.uint16)):
        bytes_per_px = bits // 8
        for k in range(1, num_channels + 1):
            shape = (height, width) if k == 1 else (k, height, width)
            axes = "YX" if k == 1 else "CYX"
            dummy = np.zeros(shape, dtype=dt)
            meta_dict: Dict[str, Any] = {"axes": axes, "mode": "composite"}
            if resolution is not None:
                meta_dict["unit"] = "um"

            buf = io.BytesIO()
            if resolution is not None:
                tifffile.imwrite(
                    buf,
                    dummy,
                    imagej=True,
                    photometric="minisblack",
                    resolution=resolution,
                    metadata=meta_dict,
                )
            else:
                tifffile.imwrite(
                    buf,
                    dummy,
                    imagej=True,
                    photometric="minisblack",
                    metadata=meta_dict,
                )
            full_bytes = buf.getvalue()
            with tifffile.TiffFile(io.BytesIO(full_bytes)) as tf:
                offset0 = int(tf.pages[0].dataoffsets[0])
            raw_byte_len = k * height * width * bytes_per_px
            prefix_b64 = base64.b64encode(full_bytes[:offset0]).decode("ascii")
            suffix_b64 = base64.b64encode(full_bytes[offset0 + raw_byte_len :]).decode("ascii")
            templates[f"{bits}_{k}"] = {
                "prefixB64": prefix_b64,
                "suffixB64": suffix_b64,
            }
    return templates


def build_legacy_studio_html(
    group: ImageGroup,
    raw_channels: Dict[str, np.ndarray],
    scalebar_cfg: ScalebarConfig,
    initial_selected_channels: Optional[List[str]] = None,
) -> str:
    """
    Build the self-contained dark-themed Real-Time HTML5/Canvas Studio for the active ImageGroup.
    """
    ordered_channels = [ch for ch in ALLOWED_CHANNELS if ch in raw_channels]
    num_channels = len(ordered_channels)
    if num_channels == 0:
        return "<div style='color:white;padding:16px;'>No valid channels in this group.</div>"

    first_arr = raw_channels[ordered_channels[0]]
    height, width = int(first_arr.shape[0]), int(first_arr.shape[1])

    channels_payload: List[Dict[str, Any]] = []
    for c_idx, ch_key in enumerate(ordered_channels):
        raw_plane = raw_channels[ch_key]

        # If a secondary channel has different dimensions, resize a visual copy for the canvas
        # while keeping raw_plane untouched for raw analysis
        if raw_plane.shape != (height, width):
            from PIL import Image
            vis_float = raw_plane.astype(np.float32)
            p_max_tmp = max(float(np.max(vis_float)), 1.0)
            pil_tmp = Image.fromarray(np.clip(vis_float / p_max_tmp * 255.0, 0, 255).astype(np.uint8))
            pil_resized = pil_tmp.resize((width, height), resample=Image.Resampling.BILINEAR)
            vis_plane = np.asarray(pil_resized, dtype=np.float32) / 255.0 * p_max_tmp
        else:
            vis_plane = raw_plane.astype(np.float32, copy=False)

        p_min = float(np.min(vis_plane)) if vis_plane.size > 0 else 0.0
        p_max = float(np.max(vis_plane)) if vis_plane.size > 0 else 255.0
        p2, p998 = np.percentile(vis_plane, [0.5, 99.5]) if vis_plane.size > 0 else (0.0, 255.0)
        if p998 <= p2:
            p2, p998 = p_min, max(p_max, p_min + 1.0)

        is_16bit = raw_plane.dtype == np.uint16 or p_max > 255.0
        b_depth = 16 if is_16bit else 8
        theoretical_max = 65535.0 if is_16bit else 255.0
        if is_16bit and 0 < p_max <= 4095.0:
            theoretical_max = 4095.0
        slider_max = max(p_max, min(theoretical_max, max(p_max * 1.15, 255.0)), 1.0)

        # Visual presentation uint16 lookup plane
        if p_min >= 0.0 and slider_max <= 65535.0:
            scale_to_u16 = 1.0
            offset_to_u16 = 0.0
            u16_vis_arr = np.clip(np.rint(vis_plane), 0, 65535).astype("<u2")
            u16_slider_max = int(min(65535, max(1, round(slider_max))))
        else:
            offset_to_u16 = p_min
            span = max(slider_max - p_min, 1e-6)
            scale_to_u16 = 65535.0 / span
            u16_vis_arr = np.clip(
                np.rint((vis_plane - offset_to_u16) * scale_to_u16), 0, 65535
            ).astype("<u2")
            u16_slider_max = 65535

        b64_vis_u16 = base64.b64encode(u16_vis_arr.tobytes()).decode("ascii")

        # Isolated Raw Analysis buffer (exact unmodified uint8 or uint16)
        raw_bits = 8 if raw_plane.dtype == np.uint8 else 16
        if raw_bits == 8:
            raw_exact_arr = np.asarray(raw_plane, dtype=np.uint8)
        else:
            raw_exact_arr = np.asarray(raw_plane, dtype="<u2")
        b64_raw_exact = base64.b64encode(raw_exact_arr.tobytes()).decode("ascii")

        default_preset = DEFAULT_PRESET_FOR_KEYWORD.get(ch_key, "Grayscale / White")
        default_hex = STUDIO_LUT_PRESETS[default_preset]["hex"]
        fname = group.files_by_channel[ch_key].filename if ch_key in group.files_by_channel else ch_key

        is_enabled = (
            True
            if initial_selected_channels is None
            else (ch_key in initial_selected_channels)
        )

        channels_payload.append(
            {
                "index": c_idx,
                "keyword": ch_key,
                "name": f"Channel {c_idx + 1} ({ch_key.title()})",
                "filename": fname,
                "bitDepth": b_depth,
                "rawContainerBits": raw_bits,
                "rawMin": int(np.min(raw_exact_arr)),
                "rawMax": int(np.max(raw_exact_arr)),
                "autoLow": round(float(p2), 2),
                "autoHigh": round(float(p998), 2),
                "sliderMax": round(slider_max, 2),
                "u16SliderMax": u16_slider_max,
                "scaleToU16": scale_to_u16,
                "offsetToU16": offset_to_u16,
                "defaultPreset": default_preset,
                "defaultHex": default_hex,
                "initialEnabled": is_enabled,
                "b64U16": b64_vis_u16,
                "b64RawExact": b64_raw_exact,
            }
        )

    imagej_templates = _build_tifffile_imagej_templates(
        num_channels=num_channels,
        height=height,
        width=width,
        calibration_um_per_px=scalebar_cfg.calibration_um_per_px,
    )

    pos_map = {
        "Bottom Right": "br",
        "Bottom Left": "bl",
        "Top Right": "tr",
        "Top Left": "tl",
    }

    config_json = json.dumps(
        {
            "storageKey": f"legacy_scope_v2_{group.group_id}",
            "groupPrefix": group.prefix,
            "groupId": group.group_id,
            "width": width,
            "height": height,
            "numChannels": num_channels,
            "objectiveLenses": list(OBJECTIVE_LENSES),
            "initialScaleBar": {
                "enabled": bool(scalebar_cfg.enabled),
                "objective": str(scalebar_cfg.objective_lens),
                "um": float(scalebar_cfg.length_um),
                "umPerPx": float(scalebar_cfg.calibration_um_per_px),
                "pos": pos_map.get(scalebar_cfg.position, "br"),
                "sizeScale": 100,
                "showBg": bool(scalebar_cfg.show_backing_box),
            },
            "imagejTemplates": imagej_templates,
            "presets": STUDIO_LUT_PRESETS,
            "channels": channels_payload,
        }
    )

    ch_summary_str = ", ".join(ch.title() for ch in ordered_channels)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  :root {{
    --bg-dark: #0e1117;
    --bg-panel: #161b22;
    --bg-card: #1f2630;
    --border-col: #30363d;
    --text-main: #f0f6fc;
    --text-muted: #8b949e;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
  }}
  * {{
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  body {{
    margin: 0;
    padding: 8px;
    background: var(--bg-dark);
    color: var(--text-main);
    overflow-x: hidden;
  }}
  .studio-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 430px;
    gap: 16px;
    align-items: start;
  }}
  .studio-grid.cinema-mode {{
    grid-template-columns: 1fr;
  }}
  @media (max-width: 1100px) {{
    .studio-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  .viewport-panel {{
    background: var(--bg-panel);
    border: 1px solid var(--border-col);
    border-radius: 10px;
    padding: 12px;
  }}
  .viewport-toolbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-col);
  }}
  .meta-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }}
  .badge {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    padding: 4px 9px;
    border-radius: 6px;
    font-size: 12px;
    color: var(--text-main);
  }}
  .badge-highlight {{
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.5);
    color: #93c5fd;
    font-weight: 600;
  }}
  .zoom-controls {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    flex-wrap: wrap;
  }}
  .canvas-stage {{
    width: 100%;
    min-height: 540px;
    max-height: 82vh;
    overflow: auto;
    background: #05070a;
    border: 1px solid #22272e;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    position: relative;
  }}
  #main-canvas {{
    display: block;
    background: #000000;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.65);
    image-rendering: auto;
  }}
  #main-canvas.pixelated {{
    image-rendering: pixelated;
  }}
  .strip-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    margin-bottom: 8px;
  }}
  .strip-title {{
    font-size: 14px;
    font-weight: 600;
  }}
  .channel-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
    gap: 10px;
  }}
  .ch-thumb-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .ch-thumb-card.muted {{
    opacity: 0.48;
  }}
  .ch-thumb-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
  }}
  .ch-thumb-sub {{
    font-size: 10.5px;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .ch-thumb-canvas {{
    width: 100%;
    height: auto;
    background: #000;
    border-radius: 4px;
    border: 1px solid #222;
  }}
  .dock-panel {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .panel-box {{
    background: var(--bg-panel);
    border: 1px solid var(--border-col);
    border-radius: 10px;
    overflow: hidden;
  }}
  .panel-header {{
    padding: 11px 13px;
    font-size: 14px;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
    background: rgba(255, 255, 255, 0.02);
    transition: background 0.15s ease;
  }}
  .panel-header:hover {{
    background: rgba(255, 255, 255, 0.06);
  }}
  .panel-header-left {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .chevron {{
    display: inline-block;
    font-size: 11px;
    color: var(--text-muted);
    transition: transform 0.18s ease;
  }}
  .panel-box.collapsed .chevron {{
    transform: rotate(-90deg);
  }}
  .panel-body {{
    padding: 12px;
    border-top: 1px solid var(--border-col);
  }}
  .panel-box.collapsed .panel-body {{
    display: none;
  }}
  .export-subbox {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
  }}
  .export-subbox-analysis {{
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.05);
  }}
  .export-subbox-pres {{
    border-color: rgba(59, 130, 246, 0.45);
    background: rgba(59, 130, 246, 0.05);
  }}
  .subbox-title {{
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .subbox-desc {{
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 8px;
    line-height: 1.35;
  }}
  .btn {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.12s ease;
  }}
  .btn:hover:not(:disabled) {{
    background: #2b3441;
    border-color: #485260;
  }}
  .btn:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}
  .btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }}
  .btn-primary {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
    padding: 8px 12px;
    font-size: 13px;
  }}
  .btn-primary:hover {{
    background: var(--accent-hover);
  }}
  .btn-success {{
    background: #10b981;
    border-color: #059669;
    color: #fff;
    font-weight: 600;
    padding: 8px 12px;
    font-size: 13px;
  }}
  .btn-success:hover {{
    background: #059669;
  }}
  .btn-row {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }}
  .merge-list {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 8px;
  }}
  .merge-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
  }}
  .merge-item-left {{
    display: flex;
    align-items: center;
    gap: 7px;
    cursor: pointer;
  }}
  .color-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.4);
  }}
  .form-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12.5px;
  }}
  .num-input, .text-input, .select-input {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12.5px;
  }}
  .num-input {{
    width: 105px;
    text-align: right;
  }}
  .text-input {{
    width: 100%;
  }}
  .select-input {{
    width: 100%;
  }}
  .cal-info-box {{
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 11.5px;
    color: #bfdbfe;
    margin-bottom: 9px;
    line-height: 1.45;
  }}
  .hw-warning-box {{
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.38);
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 11px;
    color: #fde68a;
    margin-bottom: 9px;
    line-height: 1.4;
  }}
  .ch-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    margin-bottom: 10px;
    overflow: hidden;
  }}
  .ch-card-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
    user-select: none;
  }}
  .ch-card-hdr:hover {{
    background: rgba(255, 255, 255, 0.05);
  }}
  .ch-card-title {{
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 7px;
  }}
  .ch-card-body {{
    padding: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
  }}
  .ch-card.collapsed .ch-card-body {{
    display: none;
  }}
  .ch-card.collapsed .ch-chevron {{
    transform: rotate(-90deg);
  }}
  .ch-chevron {{
    display: inline-block;
    font-size: 10px;
    color: var(--text-muted);
    transition: transform 0.15s ease;
  }}
  .slider-group {{
    margin-bottom: 7px;
  }}
  .slider-hdr {{
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 2px;
  }}
  .slider-val {{
    color: var(--text-main);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  input[type="range"] {{
    width: 100%;
    margin: 2px 0;
    accent-color: var(--accent);
    cursor: pointer;
  }}
  .two-col-sliders {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  #hist-canvas {{
    width: 100%;
    height: 92px;
    background: #0b0e14;
    border: 1px solid var(--border-col);
    border-radius: 6px;
    display: block;
  }}
  .preset-info {{
    font-size: 11px;
    color: var(--text-muted);
    margin: 4px 0 8px 0;
  }}
</style>
</head>
<body>

<div class="studio-grid" id="studio-grid">
  <!-- LEFT: LARGE MAIN PICTURE VIEWPORT + INDIVIDUAL CHANNELS -->
  <div class="viewport-panel">
    <div class="viewport-toolbar">
      <div class="meta-badges">
        <span class="badge" style="background:#1e293b; border-color:#3b82f6; color:#f8fafc; font-weight:700; font-size:13px;">🔬 FluorEasy</span>
        <span class="badge badge-highlight" id="res-badge">Original Size: {width} × {height} px (Full Quality)</span>
        <span class="badge">{group.group_id}</span>
        <span class="badge">{num_channels} Ch ({ch_summary_str})</span>
      </div>
      <div class="zoom-controls">
        <button class="btn" id="btn-undo-top" onclick="undoLastAction()" disabled title="Undo last change (Ctrl+Z or Cmd+Z)">↶ Undo</button>
        <button class="btn" id="btn-redo-top" onclick="redoLastAction()" disabled title="Redo change (Ctrl+Y or Cmd+Shift+Z)">↷ Redo</button>
        <span style="margin-left:4px;">View Size:</span>
        <button class="btn active" id="btn-fit" onclick="setZoomMode('fit')">Fit Large</button>
        <button class="btn" id="btn-fill" onclick="setZoomMode('fill')">Fill Width</button>
        <button class="btn" id="btn-1to1" onclick="setZoomMode('1to1')">100% Original Pixels (1:1)</button>
        <input type="range" id="zoom-slider" min="25" max="300" value="100" style="width:85px;" oninput="onCustomZoom(this.value)" />
        <span id="zoom-label" style="min-width:38px; text-align:right;">Fit</span>
        <button class="btn" id="btn-cinema" onclick="toggleCinemaMode()" title="Expand picture across full window width">⛶ Expand Screen</button>
      </div>
    </div>

    <div class="canvas-stage" id="canvas-stage">
      <canvas id="main-canvas" width="{width}" height="{height}"></canvas>
    </div>

    <!-- INDIVIDUAL CHANNELS STRIP -->
    <div class="strip-header">
      <div class="strip-title">🧪 Individual Channels (Preview & Direct Export)</div>
      <div class="btn-row">
        <button class="btn active" id="strip-mode-color" onclick="setStripMode('color')">Pseudocolor</button>
        <button class="btn" id="strip-mode-gray" onclick="setStripMode('gray')">Grayscale (High Contrast)</button>
      </div>
    </div>
    <div class="channel-strip" id="channel-strip"></div>
  </div>

  <!-- RIGHT: COLLAPSIBLE STUDIO DOCK -->
  <div class="dock-panel">

    <!-- PANEL 1: EXPORT: ANALYSIS (RAW) & PRESENTATION (VISUAL) -->
    <div class="panel-box" id="panel-merge">
      <div class="panel-header" onclick="togglePanel('panel-merge')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>💾 1. Export: Analysis (Raw) & Presentation (Visual)</span>
        </div>
      </div>
      <div class="panel-body">
        <div class="form-row" style="margin-bottom:10px;">
          <label style="font-size:12px; color:var(--text-muted); white-space:nowrap;">File Name Prefix:</label>
          <input type="text" id="export-filename" class="text-input" value="{group.prefix}" />
        </div>

        <!-- SECTION 1A: SAVE FOR ANALYSIS (RAW DATA) -->
        <div class="export-subbox export-subbox-analysis">
          <div class="subbox-title">
            <span style="color:#34d399;">🔬 1A. Save for Analysis (Raw Data)</span>
            <div class="btn-row">
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllAnalysisChannels(true)">All</button>
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllAnalysisChannels(false)">None</button>
            </div>
          </div>
          <div class="subbox-desc">
            Select channels to include. Exports a <b>raw, multi-dimensional scientific TIFF</b> (<code>axes='CYX'</code> via <code>tifffile</code>) preserving original grayscale intensity data (8/16-bit). No LUTs, no flattening.
          </div>
          <div class="merge-list" id="analysis-checklist"></div>
          <button class="btn btn-success" style="width:100%;" onclick="downloadRawAnalysisTIFF()">
            ⬇️ Save for Analysis (Raw Multi-D TIFF)
          </button>
        </div>

        <!-- SECTION 1B: SAVE FOR PRESENTATION (VISUAL RENDER) -->
        <div class="export-subbox export-subbox-pres" style="margin-bottom:0;">
          <div class="subbox-title">
            <span style="color:#60a5fa;">🖼️ 1B. Save for Presentation (Visual Render)</span>
            <div class="btn-row">
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllChannelsMerged(true)">All On</button>
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllChannelsMerged(false)">All Off</button>
            </div>
          </div>
          <div class="subbox-desc">
            Exports the visually pseudocolored and merged image <b>with the scalebar burned in</b> (8-bit RGB, {width}×{height} px).
          </div>
          <div class="merge-list" id="merge-checklist"></div>

          <div class="form-row" style="margin-top:8px;">
            <span style="font-size:12px; font-weight:600;">Output Format:</span>
            <select id="pres-format-select" class="select-input" style="width:185px;">
              <option value="PNG">PNG (Lossless 8-bit RGB)</option>
              <option value="JPEG">JPEG (8-bit RGB)</option>
              <option value="TIFF">8-bit RGB TIFF</option>
            </select>
          </div>

          <div class="btn-row" style="margin-top:8px;">
            <button class="btn btn-primary" style="flex:1;" onclick="downloadPresentationImage()">
              ⬇️ Save for Presentation
            </button>
            <button class="btn" onclick="downloadAllSelectedChannelsPresentation()" title="Save each visible channel + merged composite">
              ⬇️ Save Each Ch + Merge
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- PANEL 2: COLOR ADJUSTMENT -->
    <div class="panel-box" id="panel-color">
      <div class="panel-header" onclick="togglePanel('panel-color')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>🎨 2. Color Adjustment</span>
        </div>
        <div class="btn-row" onclick="event.stopPropagation();">
          <button class="btn" id="btn-undo-panel" onclick="undoLastAction()" disabled title="Undo (Ctrl+Z or Cmd+Z)">↶ Undo</button>
          <button class="btn" onclick="autoContrastAll()">✨ Auto All</button>
          <button class="btn" onclick="resetAllChannels()">↺ Reset All</button>
        </div>
      </div>
      <div class="panel-body">
        <canvas id="hist-canvas" width="380" height="92"></canvas>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px; margin-bottom:8px;">
          Drag any slider below — updates live at 60 FPS. Press <b>Ctrl+Z</b> (or <b>Cmd+Z</b>) to undo.
        </div>
        <div id="channels-container"></div>
      </div>
    </div>

    <!-- PANEL 3: SCALEBAR PANEL -->
    <div class="panel-box" id="panel-scalebar">
      <div class="panel-header" onclick="togglePanel('panel-scalebar')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>📏 3. Scalebar Panel</span>
        </div>
        <label style="font-size:12px; font-weight:normal; cursor:pointer; display:flex; align-items:center; gap:5px;" onclick="event.stopPropagation();">
          <input type="checkbox" id="sb-enabled" checked onchange="onScaleBarChange()" /> Show
        </label>
      </div>
      <div class="panel-body">
        <div class="cal-info-box" id="cal-info-box"></div>

        <div class="form-row">
          <span>Objective Lens:</span>
          <select id="sb-obj-select" class="select-input" style="width:140px;" onchange="selectObjective(this.value)">
            <option value="4x">4x</option>
            <option value="10x">10x</option>
            <option value="16x">16x</option>
            <option value="20x">20x</option>
            <option value="40x">40x</option>
            <option value="100x">100x</option>
          </select>
        </div>

        <div class="btn-row" style="margin-bottom:8px; justify-content:flex-end;">
          <span style="font-size:11px; color:var(--text-muted); align-self:center; margin-right:auto;">Quick Lens:</span>
          <button class="btn" id="obj-4x" onclick="selectObjective('4x')">4x</button>
          <button class="btn" id="obj-10x" onclick="selectObjective('10x')">10x</button>
          <button class="btn" id="obj-16x" onclick="selectObjective('16x')">16x</button>
          <button class="btn" id="obj-20x" onclick="selectObjective('20x')">20x</button>
          <button class="btn" id="obj-40x" onclick="selectObjective('40x')">40x</button>
          <button class="btn" id="obj-100x" onclick="selectObjective('100x')">100x</button>
        </div>

        <div class="form-row">
          <span>Scalebar Length (μm):</span>
          <div style="display:flex; align-items:center; gap:5px;">
            <input type="number" id="sb-um" class="num-input" min="0.5" max="5000" step="any" value="25" oninput="onScaleBarChange()" />
            <span style="font-size:12px; font-weight:600;">μm</span>
          </div>
        </div>

        <div class="btn-row" style="margin-bottom:8px; justify-content:flex-end;">
          <span style="font-size:11px; color:var(--text-muted); align-self:center; margin-right:auto;">Quick μm:</span>
          <button class="btn" onclick="setQuickUm(10)">10 μm</button>
          <button class="btn" onclick="setQuickUm(25)">25 μm</button>
          <button class="btn" onclick="setQuickUm(50)">50 μm</button>
          <button class="btn" onclick="setQuickUm(100)">100 μm</button>
        </div>

        <div class="form-row">
          <span title="Hardware-specific Calibration Factor in micrometers per pixel (μm/pixel)">Calibration Factor (μm/pixel):</span>
          <div style="display:flex; gap:5px; align-items:center;">
            <input type="number" id="sb-um-per-px" class="num-input" min="0.0001" max="100" step="0.005" value="0.1625" oninput="onCustomCalibrationInput()" />
          </div>
        </div>

        <div class="form-row">
          <span>Corner & Text Size:</span>
          <div style="display:flex; gap:6px; align-items:center;">
            <select id="sb-pos" class="select-input" style="width:130px;" onchange="onScaleBarChange()">
              <option value="br">Bottom Right</option>
              <option value="bl">Bottom Left</option>
              <option value="tr">Top Right</option>
              <option value="tl">Top Left</option>
            </select>
            <input type="range" id="sb-thickness" min="50" max="250" value="100" style="width:75px;" title="Scalebar & text size" oninput="onScaleBarChange()" />
          </div>
        </div>

        <div class="form-row" style="margin-bottom:0;">
          <span>Scalebar Background:</span>
          <label style="font-size:12.5px; cursor:pointer; display:flex; align-items:center; gap:6px;">
            <input type="checkbox" id="sb-bg" checked onchange="onScaleBarChange()" />
            <span>Show Background Box</span>
          </label>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const CONFIG = {config_json};

const visPlanesU16 = [];
const rawAnalysisBuffers = [];

for (let c = 0; c < CONFIG.numChannels; c++) {{
  const b64Vis = CONFIG.channels[c].b64U16;
  const binVis = atob(b64Vis);
  const bytesVis = new Uint8Array(binVis.length);
  for (let i = 0; i < binVis.length; i++) {{
    bytesVis[i] = binVis.charCodeAt(i);
  }}
  visPlanesU16.push(new Uint16Array(bytesVis.buffer));

  const b64Raw = CONFIG.channels[c].b64RawExact;
  const binRaw = atob(b64Raw);
  const bytesRaw = new Uint8Array(binRaw.length);
  for (let i = 0; i < binRaw.length; i++) {{
    bytesRaw[i] = binRaw.charCodeAt(i);
  }}
  rawAnalysisBuffers.push(Object.freeze({{
    bits: CONFIG.channels[c].rawContainerBits,
    byteView: bytesRaw,
    typedView: CONFIG.channels[c].rawContainerBits === 8
      ? bytesRaw
      : new Uint16Array(bytesRaw.buffer)
  }}));
}}

const lutNorm = [];
for (let c = 0; c < CONFIG.numChannels; c++) {{
  lutNorm.push(new Float32Array(65536));
}}

let state = {{
  zoomMode: 'fit',
  customZoom: 100,
  cinemaMode: false,
  stripMode: 'color',
  analysisSelected: CONFIG.channels.map(ch => Boolean(ch.initialEnabled)),
  scaleBar: {{
    enabled: CONFIG.initialScaleBar.enabled,
    objective: CONFIG.initialScaleBar.objective || '40x',
    um: CONFIG.initialScaleBar.um || 25,
    umPerPx: CONFIG.initialScaleBar.umPerPx || 0.1625,
    pos: CONFIG.initialScaleBar.pos || 'br',
    sizeScale: CONFIG.initialScaleBar.sizeScale || 100,
    showBg: CONFIG.initialScaleBar.showBg !== false
  }},
  channels: []
}};

function createDefaultChannelState(c) {{
  const chMeta = CONFIG.channels[c];
  return {{
    enabled: Boolean(chMeta.initialEnabled),
    preset: chMeta.defaultPreset,
    hex: chMeta.defaultHex,
    brightness: 0,
    contrast: 0,
    exposure: 0,
    gamma: 1.0,
    blackPoint: chMeta.rawMin,
    whitePoint: chMeta.rawMax,
    invert: false
  }};
}}

function initStudioState() {{
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.channels.push(createDefaultChannelState(c));
  }}
}}

// ============================================================================
// UNDO (Ctrl+Z / Cmd+Z) & REDO (Ctrl+Y / Cmd+Shift+Z) HISTORY STACK
// ============================================================================
const undoStack = [];
const redoStack = [];
let activeDragSnapshot = null;

function getStateSnapshot() {{
  return JSON.stringify({{
    analysisSelected: state.analysisSelected,
    scaleBar: state.scaleBar,
    channels: state.channels
  }});
}}

function beginContinuousEdit() {{
  if (activeDragSnapshot === null) {{
    activeDragSnapshot = getStateSnapshot();
  }}
}}

function commitContinuousEdit() {{
  if (activeDragSnapshot !== null) {{
    pushUndoSnapshot(activeDragSnapshot);
    activeDragSnapshot = null;
  }}
}}

function recordDiscreteEdit() {{
  commitContinuousEdit();
  pushUndoSnapshot(getStateSnapshot());
}}

function pushUndoSnapshot(snapStr) {{
  if (!snapStr || snapStr === getStateSnapshot()) return;
  if (undoStack.length > 0 && undoStack[undoStack.length - 1] === snapStr) return;
  undoStack.push(snapStr);
  if (undoStack.length > 60) undoStack.shift();
  redoStack.length = 0;
  updateUndoRedoButtons();
}}

function updateUndoRedoButtons() {{
  const canUndo = undoStack.length > 0 || activeDragSnapshot !== null;
  const canRedo = redoStack.length > 0;
  const uTop = document.getElementById("btn-undo-top");
  const rTop = document.getElementById("btn-redo-top");
  const uPanel = document.getElementById("btn-undo-panel");
  if (uTop) uTop.disabled = !canUndo;
  if (rTop) rTop.disabled = !canRedo;
  if (uPanel) uPanel.disabled = !canUndo;
}}

function applyStateSnapshot(snapStr) {{
  try {{
    const snap = JSON.parse(snapStr);
    if (Array.isArray(snap.analysisSelected)) {{
      state.analysisSelected = snap.analysisSelected.map(Boolean);
    }}
    if (snap.scaleBar) {{
      state.scaleBar = {{ ...state.scaleBar, ...snap.scaleBar }};
    }}
    if (Array.isArray(snap.channels)) {{
      state.channels = snap.channels.map(ch => ({{ ...ch }}));
    }}
    syncScaleBarUI();
    for (let c = 0; c < CONFIG.numChannels; c++) {{
      const aChk = document.getElementById(`analysis-chk-${{c}}`);
      if (aChk) aChk.checked = state.analysisSelected[c];
      const tCard = document.getElementById(`thumb-card-${{c}}`);
      if (tCard) tCard.className = "ch-thumb-card" + (state.channels[c].enabled ? "" : " muted");
      syncChannelSlidersDOM(c);
      updateChannelLUT(c);
    }}
    requestRender();
  }} catch (e) {{
    console.warn("Failed to apply undo/redo snapshot:", e);
  }}
}}

function undoLastAction() {{
  if (activeDragSnapshot !== null) {{
    pushUndoSnapshot(activeDragSnapshot);
    activeDragSnapshot = null;
  }}
  if (undoStack.length === 0) return;
  const currentSnap = getStateSnapshot();
  const prevSnap = undoStack.pop();
  redoStack.push(currentSnap);
  applyStateSnapshot(prevSnap);
  updateUndoRedoButtons();
}}

function redoLastAction() {{
  commitContinuousEdit();
  if (redoStack.length === 0) return;
  const currentSnap = getStateSnapshot();
  const nextSnap = redoStack.pop();
  undoStack.push(currentSnap);
  applyStateSnapshot(nextSnap);
  updateUndoRedoButtons();
}}

function handleGlobalKeyDown(e) {{
  const activeTag = document.activeElement ? document.activeElement.id : "";
  if (activeTag === "export-filename") return;

  const isCtrlOrCmd = e.ctrlKey || e.metaKey;
  if (!isCtrlOrCmd) return;

  const key = (e.key || "").toLowerCase();
  if (key === "z" && !e.shiftKey) {{
    if (undoStack.length > 0 || activeDragSnapshot !== null) {{
      e.preventDefault();
      undoLastAction();
    }} else if (window.parent && window.parent.document) {{
      const stUndo = window.parent.document.querySelector('.st-key-hidden_undo_trigger button');
      if (stUndo && !stUndo.disabled) {{
        e.preventDefault();
        stUndo.click();
      }}
    }}
  }} else if ((key === "z" && e.shiftKey) || key === "y") {{
    if (redoStack.length > 0) {{
      e.preventDefault();
      redoLastAction();
    }} else if (window.parent && window.parent.document) {{
      const stRedo = window.parent.document.querySelector('.st-key-hidden_redo_trigger button');
      if (stRedo && !stRedo.disabled) {{
        e.preventDefault();
        stRedo.click();
      }}
    }}
  }}
}}

function togglePanel(panelId) {{
  const el = document.getElementById(panelId);
  if (el) el.classList.toggle("collapsed");
}}

function toggleChannelCard(c) {{
  const el = document.getElementById(`ch-card-${{c}}`);
  if (el) el.classList.toggle("collapsed");
}}

function hexToRgbNorm(hex) {{
  let clean = (hex || "#ffffff").replace("#", "").trim();
  if (clean.length === 3) clean = clean[0]+clean[0]+clean[1]+clean[1]+clean[2]+clean[2];
  const num = parseInt(clean, 16);
  if (isNaN(num)) return [1, 1, 1];
  return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255];
}}

function rawToU16(c, rawVal) {{
  const m = CONFIG.channels[c];
  return Math.max(0, Math.min(65535, Math.round((rawVal - m.offsetToU16) * m.scaleToU16)));
}}

function updateChannelLUT(c) {{
  const cfg = state.channels[c];
  const m = CONFIG.channels[c];
  const lut = lutNorm[c];
  const maxU16 = m.u16SliderMax;

  const bpU16 = rawToU16(c, cfg.blackPoint);
  const wpU16 = rawToU16(c, cfg.whitePoint);
  const denom = Math.max(1e-6, wpU16 - bpU16);

  const expFactor = Math.abs(cfg.exposure) > 1e-4 ? Math.pow(2.0, cfg.exposure) : 1.0;
  const hasBC = Math.abs(cfg.contrast) > 1e-4 || Math.abs(cfg.brightness) > 1e-4;
  const cNorm = Math.max(-0.98, Math.min(0.98, cfg.contrast / 100.0));
  const contrastFactor = Math.tan((cNorm + 1.0) * (Math.PI / 4.0));
  const bOffset = (cfg.brightness / 100.0) * 0.5;
  const invGamma = 1.0 / Math.max(0.05, cfg.gamma);
  const hasGamma = Math.abs(cfg.gamma - 1.0) > 1e-4;
  const invert = cfg.invert;

  for (let v = 0; v <= maxU16; v++) {{
    let norm = (v - bpU16) / denom;
    if (norm < 0.0) norm = 0.0;
    else if (norm > 1.0) norm = 1.0;

    if (expFactor !== 1.0) norm *= expFactor;
    if (hasBC) norm = contrastFactor * (norm - 0.5 + bOffset) + 0.5;
    if (norm < 0.0) norm = 0.0;
    else if (norm > 1.0) norm = 1.0;

    if (hasGamma) norm = Math.pow(norm, invGamma);
    if (invert) norm = 1.0 - norm;
    lut[v] = norm;
  }}
}}

function buildUI() {{
  syncScaleBarUI();
  buildAnalysisChecklist();
  buildMergeChecklist();
  buildChannelStripCards();
  buildChannelControlCards();
}}

function buildAnalysisChecklist() {{
  const container = document.getElementById("analysis-checklist");
  container.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const isSel = state.analysisSelected[c];
    const row = document.createElement("div");
    row.className = "merge-item";
    row.innerHTML = `
      <label class="merge-item-left" title="${{chMeta.filename}}">
        <input type="checkbox" id="analysis-chk-${{c}}" ${{isSel ? 'checked' : ''}} onchange="toggleAnalysisChannel(${{c}}, this.checked)" />
        <span style="font-weight:600;">${{chMeta.name}}</span>
      </label>
      <span style="font-size:11px; color:#34d399; font-variant-numeric:tabular-nums;">
        Raw ${{chMeta.bitDepth}}-bit [${{chMeta.rawMin}}–${{chMeta.rawMax}}]
      </span>
    `;
    container.appendChild(row);
  }}
}}

function toggleAnalysisChannel(c, checked) {{
  recordDiscreteEdit();
  state.analysisSelected[c] = checked;
  const el = document.getElementById(`analysis-chk-${{c}}`);
  if (el) el.checked = checked;
}}

function setAllAnalysisChannels(checked) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.analysisSelected[c] = checked;
    const el = document.getElementById(`analysis-chk-${{c}}`);
    if (el) el.checked = checked;
  }}
}}

function buildMergeChecklist() {{
  const container = document.getElementById("merge-checklist");
  container.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    const row = document.createElement("div");
    row.className = "merge-item";
    row.innerHTML = `
      <label class="merge-item-left" title="${{chMeta.filename}}">
        <input type="checkbox" id="merge-chk-${{c}}" ${{cfg.enabled ? 'checked' : ''}} onchange="toggleChannelEnabled(${{c}}, this.checked)" />
        <span class="color-dot" id="merge-dot-${{c}}" style="background:${{cfg.hex}};"></span>
        <span style="font-weight:600;">${{chMeta.keyword.toUpperCase()}}:</span>
        <span id="merge-preset-lbl-${{c}}" style="color:var(--text-muted);">${{cfg.preset}}</span>
      </label>
      <button class="btn" style="padding:2px 7px; font-size:11px;" onclick="soloChannel(${{c}})" title="Show only this channel">Solo</button>
    `;
    container.appendChild(row);
  }}
}}

function buildChannelStripCards() {{
  const strip = document.getElementById("channel-strip");
  strip.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    const card = document.createElement("div");
    card.className = "ch-thumb-card" + (cfg.enabled ? "" : " muted");
    card.id = `thumb-card-${{c}}`;
    card.innerHTML = `
      <div class="ch-thumb-hdr">
        <span><span class="color-dot" id="thumb-dot-${{c}}" style="background:${{cfg.hex}}; margin-right:5px;"></span>${{chMeta.keyword.toUpperCase()}}: <span id="thumb-title-${{c}}">${{cfg.preset}}</span></span>
        <div class="btn-row">
          <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="downloadSingleChannelRawTIFF(${{c}})" title="Save Raw Unmodified Analysis TIFF of ${{chMeta.name}}">🔬 Raw</button>
          <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="downloadSingleChannelPresentation(${{c}})" title="Save Visual Presentation Image of ${{chMeta.name}}">🖼️ Visual</button>
        </div>
      </div>
      <div class="ch-thumb-sub" title="${{chMeta.filename}}">📄 ${{chMeta.filename}}</div>
      <canvas id="thumb-canvas-${{c}}" class="ch-thumb-canvas" width="{width}" height="{height}"></canvas>
    `;
    strip.appendChild(card);
  }}
}}

function buildChannelControlCards() {{
  const container = document.getElementById("channels-container");
  container.innerHTML = "";
  const presetNames = Object.keys(CONFIG.presets);

  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    const stepVal = Math.max(0.1, Number(((chMeta.sliderMax) / 500).toFixed(2)));

    const optionsHtml = presetNames.map(p =>
      `<option value="${{p}}" ${{p === cfg.preset ? 'selected' : ''}}>${{p}}</option>`
    ).join("");

    const card = document.createElement("div");
    card.className = "ch-card";
    card.id = `ch-card-${{c}}`;
    card.innerHTML = `
      <div class="ch-card-hdr" onclick="toggleChannelCard(${{c}})">
        <div class="ch-card-title">
          <span class="ch-chevron">▼</span>
          <input type="checkbox" id="card-chk-${{c}}" ${{cfg.enabled ? 'checked' : ''}} onclick="event.stopPropagation();" onchange="toggleChannelEnabled(${{c}}, this.checked)" title="Include channel in merged composite" />
          <span class="color-dot" id="card-dot-${{c}}" style="background:${{cfg.hex}};"></span>
          <span>${{chMeta.name}}</span>
        </div>
        <div class="btn-row" onclick="event.stopPropagation();">
          <button class="btn" style="padding:3px 7px; font-size:11px;" onclick="autoContrastChannel(${{c}})">Auto</button>
          <button class="btn" style="padding:3px 7px; font-size:11px;" onclick="resetChannel(${{c}})">Reset</button>
        </div>
      </div>

      <div class="ch-card-body">
        <div style="display:grid; grid-template-columns: 1fr 42px; gap:6px; align-items:center; margin-bottom:4px;">
          <select id="preset-sel-${{c}}" class="select-input" onchange="onPresetSelect(${{c}}, this.value)">
            ${{optionsHtml}}
          </select>
          <input type="color" id="color-pick-${{c}}" value="${{cfg.hex}}" style="width:42px; height:29px; padding:0; border:1px solid var(--border-col); border-radius:6px; background:transparent; cursor:pointer;" oninput="onColorPick(${{c}}, this.value)" />
        </div>
        <div class="preset-info" id="preset-info-${{c}}"></div>

        <div class="two-col-sliders">
          <div class="slider-group">
            <div class="slider-hdr"><span>Brightness</span><span class="slider-val" id="val-bright-${{c}}">${{cfg.brightness}}</span></div>
            <input type="range" id="sl-bright-${{c}}" min="-100" max="100" step="1" value="${{cfg.brightness}}" oninput="onSliderInput(${{c}}, 'brightness', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Contrast</span><span class="slider-val" id="val-cont-${{c}}">${{cfg.contrast}}</span></div>
            <input type="range" id="sl-cont-${{c}}" min="-100" max="100" step="1" value="${{cfg.contrast}}" oninput="onSliderInput(${{c}}, 'contrast', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Exposure (EV)</span><span class="slider-val" id="val-exp-${{c}}">${{Number(cfg.exposure).toFixed(2)}}</span></div>
            <input type="range" id="sl-exp-${{c}}" min="-3.0" max="3.0" step="0.05" value="${{cfg.exposure}}" oninput="onSliderInput(${{c}}, 'exposure', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Gamma (γ)</span><span class="slider-val" id="val-gam-${{c}}">${{Number(cfg.gamma).toFixed(2)}}</span></div>
            <input type="range" id="sl-gam-${{c}}" min="0.10" max="3.00" step="0.05" value="${{cfg.gamma}}" oninput="onSliderInput(${{c}}, 'gamma', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Black Point</span><span class="slider-val" id="val-bp-${{c}}">${{Math.round(cfg.blackPoint)}}</span></div>
            <input type="range" id="sl-bp-${{c}}" min="0" max="${{chMeta.sliderMax}}" step="${{stepVal}}" value="${{cfg.blackPoint}}" oninput="onSliderInput(${{c}}, 'blackPoint', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>White Point</span><span class="slider-val" id="val-wp-${{c}}">${{Math.round(cfg.whitePoint)}}</span></div>
            <input type="range" id="sl-wp-${{c}}" min="0" max="${{chMeta.sliderMax}}" step="${{stepVal}}" value="${{cfg.whitePoint}}" oninput="onSliderInput(${{c}}, 'whitePoint', this.value)" />
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:11.5px; color:var(--text-muted); margin-top:2px;">
          <span>Raw: [${{chMeta.rawMin}} – ${{chMeta.rawMax}}] (${{chMeta.bitDepth}}-bit) • ${{chMeta.filename}}</span>
          <label style="cursor:pointer;"><input type="checkbox" id="chk-inv-${{c}}" ${{cfg.invert ? 'checked' : ''}} onchange="onInvertToggle(${{c}}, this.checked)" /> Invert</label>
        </div>
      </div>
    `;
    container.appendChild(card);
    updatePresetDescription(c);
  }}
}}

function updatePresetDescription(c) {{
  const cfg = state.channels[c];
  const p = CONFIG.presets[cfg.preset];
  const el = document.getElementById(`preset-info-${{c}}`);
  if (!el) return;
  if (p && (p.ex || p.em)) {{
    el.innerHTML = `<span style="color:${{cfg.hex}}; font-weight:600;">Ex ${{p.ex}} nm / Em ${{p.em}} nm</span> — ${{p.desc}}`;
  }} else if (p) {{
    el.textContent = p.desc;
  }} else {{
    el.textContent = "";
  }}
}}

let renderPending = false;
function requestRender() {{
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {{
    renderPending = false;
    renderAll();
  }});
}}

function onSliderInput(c, field, val) {{
  beginContinuousEdit();
  const num = parseFloat(val);
  const cfg = state.channels[c];
  if (field === 'blackPoint') {{
    cfg.blackPoint = Math.min(num, cfg.whitePoint - 0.5);
    document.getElementById(`val-bp-${{c}}`).textContent = Math.round(cfg.blackPoint);
  }} else if (field === 'whitePoint') {{
    cfg.whitePoint = Math.max(num, cfg.blackPoint + 0.5);
    document.getElementById(`val-wp-${{c}}`).textContent = Math.round(cfg.whitePoint);
  }} else if (field === 'brightness') {{
    cfg.brightness = Math.round(num);
    document.getElementById(`val-bright-${{c}}`).textContent = cfg.brightness;
  }} else if (field === 'contrast') {{
    cfg.contrast = Math.round(num);
    document.getElementById(`val-cont-${{c}}`).textContent = cfg.contrast;
  }} else if (field === 'exposure') {{
    cfg.exposure = num;
    document.getElementById(`val-exp-${{c}}`).textContent = num.toFixed(2);
  }} else if (field === 'gamma') {{
    cfg.gamma = num;
    document.getElementById(`val-gam-${{c}}`).textContent = num.toFixed(2);
  }}
  updateChannelLUT(c);
  updateUndoRedoButtons();
  requestRender();
}}

function onInvertToggle(c, checked) {{
  recordDiscreteEdit();
  state.channels[c].invert = checked;
  updateChannelLUT(c);
  requestRender();
}}

function onPresetSelect(c, presetName) {{
  recordDiscreteEdit();
  const cfg = state.channels[c];
  cfg.preset = presetName;
  if (CONFIG.presets[presetName] && presetName !== "Custom Color") {{
    cfg.hex = CONFIG.presets[presetName].hex;
    document.getElementById(`color-pick-${{c}}`).value = cfg.hex;
  }}
  syncChannelColorUI(c);
  updatePresetDescription(c);
  requestRender();
}}

function onColorPick(c, hexVal) {{
  beginContinuousEdit();
  const cfg = state.channels[c];
  cfg.hex = hexVal;
  syncChannelColorUI(c);
  updateUndoRedoButtons();
  requestRender();
}}

function syncChannelColorUI(c) {{
  const cfg = state.channels[c];
  const ids = [`merge-dot-${{c}}`, `thumb-dot-${{c}}`, `card-dot-${{c}}`];
  ids.forEach(id => {{
    const el = document.getElementById(id);
    if (el) el.style.background = cfg.hex;
  }});
  const mLbl = document.getElementById(`merge-preset-lbl-${{c}}`);
  if (mLbl) mLbl.textContent = cfg.preset;
  const tLbl = document.getElementById(`thumb-title-${{c}}`);
  if (tLbl) tLbl.textContent = cfg.preset;
}}

function _setChannelEnabledSilent(c, enabled) {{
  state.channels[c].enabled = enabled;
  const mChk = document.getElementById(`merge-chk-${{c}}`);
  if (mChk) mChk.checked = enabled;
  const cChk = document.getElementById(`card-chk-${{c}}`);
  if (cChk) cChk.checked = enabled;
  const tCard = document.getElementById(`thumb-card-${{c}}`);
  if (tCard) tCard.className = "ch-thumb-card" + (enabled ? "" : " muted");
}}

function toggleChannelEnabled(c, enabled) {{
  recordDiscreteEdit();
  _setChannelEnabledSilent(c, enabled);
  requestRender();
}}

function setAllChannelsMerged(enabled) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    _setChannelEnabledSilent(c, enabled);
  }}
  requestRender();
}}

function soloChannel(targetIdx) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    _setChannelEnabledSilent(c, c === targetIdx);
  }}
  requestRender();
}}

function syncChannelSlidersDOM(c) {{
  const cfg = state.channels[c];
  document.getElementById(`card-chk-${{c}}`).checked = cfg.enabled;
  document.getElementById(`merge-chk-${{c}}`).checked = cfg.enabled;
  document.getElementById(`preset-sel-${{c}}`).value = cfg.preset;
  document.getElementById(`color-pick-${{c}}`).value = cfg.hex;
  document.getElementById(`sl-bright-${{c}}`).value = cfg.brightness;
  document.getElementById(`val-bright-${{c}}`).textContent = cfg.brightness;
  document.getElementById(`sl-cont-${{c}}`).value = cfg.contrast;
  document.getElementById(`val-cont-${{c}}`).textContent = cfg.contrast;
  document.getElementById(`sl-exp-${{c}}`).value = cfg.exposure;
  document.getElementById(`val-exp-${{c}}`).textContent = Number(cfg.exposure).toFixed(2);
  document.getElementById(`sl-gam-${{c}}`).value = cfg.gamma;
  document.getElementById(`val-gam-${{c}}`).textContent = Number(cfg.gamma).toFixed(2);
  document.getElementById(`sl-bp-${{c}}`).value = cfg.blackPoint;
  document.getElementById(`val-bp-${{c}}`).textContent = Math.round(cfg.blackPoint);
  document.getElementById(`sl-wp-${{c}}`).value = cfg.whitePoint;
  document.getElementById(`val-wp-${{c}}`).textContent = Math.round(cfg.whitePoint);
  document.getElementById(`chk-inv-${{c}}`).checked = cfg.invert;
  syncChannelColorUI(c);
  updatePresetDescription(c);
}}

function resetChannel(c) {{
  recordDiscreteEdit();
  state.channels[c] = createDefaultChannelState(c);
  syncChannelSlidersDOM(c);
  updateChannelLUT(c);
  requestRender();
}}

function resetAllChannels() {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.channels[c] = createDefaultChannelState(c);
    syncChannelSlidersDOM(c);
    updateChannelLUT(c);
  }}
  requestRender();
}}

function autoContrastChannel(c) {{
  recordDiscreteEdit();
  const chMeta = CONFIG.channels[c];
  const cfg = state.channels[c];
  cfg.blackPoint = chMeta.autoLow;
  cfg.whitePoint = chMeta.autoHigh;
  cfg.brightness = 0;
  cfg.contrast = 0;
  cfg.exposure = 0;
  cfg.gamma = 1.0;
  syncChannelSlidersDOM(c);
  updateChannelLUT(c);
  requestRender();
}}

function autoContrastAll() {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    cfg.blackPoint = chMeta.autoLow;
    cfg.whitePoint = chMeta.autoHigh;
    syncChannelSlidersDOM(c);
    updateChannelLUT(c);
  }}
  requestRender();
}}

// Objective lens & Scalebar handlers
function updateCalibrationInfoBox() {{
  const sb = state.scaleBar;
  const box = document.getElementById("cal-info-box");
  if (!box) return;
  const umPerPx = sb.umPerPx > 0 ? sb.umPerPx : 0.1625;
  const pxPerUm = 1.0 / umPerPx;
  const fovW = CONFIG.width * umPerPx;
  const fovH = CONFIG.height * umPerPx;
  const barPx = Math.round(sb.um / umPerPx);

  box.innerHTML =
    `<div><b>Active Lens (${{sb.objective}}):</b> ${{umPerPx.toFixed(4)}} μm/px (${{pxPerUm.toFixed(3)}} px/μm)</div>` +
    `<div><b>Field of View:</b> ${{fovW.toFixed(1)}} × ${{fovH.toFixed(1)}} μm (${{CONFIG.width}}×${{CONFIG.height}} px)</div>` +
    `<div><b>Scalebar:</b> ${{sb.um}} μm = <b>${{barPx}} px</b></div>`;
}}

function syncScaleBarUI() {{
  const sb = state.scaleBar;
  document.getElementById("sb-enabled").checked = sb.enabled;
  document.getElementById("sb-obj-select").value = sb.objective;
  document.getElementById("sb-um").value = sb.um;
  document.getElementById("sb-um-per-px").value = Number(sb.umPerPx).toFixed(4);
  document.getElementById("sb-pos").value = sb.pos;
  document.getElementById("sb-thickness").value = sb.sizeScale;
  const bgChk = document.getElementById("sb-bg");
  if (bgChk) bgChk.checked = sb.showBg !== false;

  ['4x', '10x', '16x', '20x', '40x', '100x'].forEach(k => {{
    const btn = document.getElementById(`obj-${{k}}`);
    if (btn) btn.className = "btn" + (sb.objective === k ? " active" : "");
  }});
  updateCalibrationInfoBox();
}}

function selectObjective(objKey) {{
  recordDiscreteEdit();
  state.scaleBar.objective = objKey;
  syncScaleBarUI();
  requestRender();
}}

function setQuickUm(umVal) {{
  recordDiscreteEdit();
  state.scaleBar.um = umVal;
  document.getElementById("sb-um").value = umVal;
  updateCalibrationInfoBox();
  requestRender();
}}

function onCustomCalibrationInput() {{
  beginContinuousEdit();
  const val = parseFloat(document.getElementById("sb-um-per-px").value);
  if (!isNaN(val) && val > 0) {{
    state.scaleBar.umPerPx = val;
    updateCalibrationInfoBox();
    updateUndoRedoButtons();
    requestRender();
  }}
}}

function onScaleBarChange() {{
  beginContinuousEdit();
  const sb = state.scaleBar;
  sb.enabled = document.getElementById("sb-enabled").checked;
  const umVal = parseFloat(document.getElementById("sb-um").value);
  if (!isNaN(umVal) && umVal > 0) sb.um = umVal;
  sb.pos = document.getElementById("sb-pos").value;
  sb.sizeScale = parseFloat(document.getElementById("sb-thickness").value) || 100;
  const bgChk = document.getElementById("sb-bg");
  if (bgChk) sb.showBg = bgChk.checked;
  updateCalibrationInfoBox();
  updateUndoRedoButtons();
  requestRender();
}}

function setZoomMode(mode) {{
  state.zoomMode = mode;
  ['fit', 'fill', '1to1'].forEach(m => {{
    const btn = document.getElementById(`btn-${{m}}`);
    if (btn) btn.className = "btn" + (mode === m ? " active" : "");
  }});
  applyCanvasDisplaySize();
}}

function onCustomZoom(val) {{
  state.zoomMode = 'custom';
  state.customZoom = parseInt(val, 10) || 100;
  ['fit', 'fill', '1to1'].forEach(m => {{
    const btn = document.getElementById(`btn-${{m}}`);
    if (btn) btn.className = "btn";
  }});
  applyCanvasDisplaySize();
}}

function toggleCinemaMode() {{
  state.cinemaMode = !state.cinemaMode;
  const grid = document.getElementById("studio-grid");
  const btn = document.getElementById("btn-cinema");
  if (state.cinemaMode) {{
    grid.classList.add("cinema-mode");
    btn.classList.add("active");
    btn.textContent = "⛶ Exit Expand";
  }} else {{
    grid.classList.remove("cinema-mode");
    btn.classList.remove("active");
    btn.textContent = "⛶ Expand Screen";
  }}
  applyCanvasDisplaySize();
}}

function applyCanvasDisplaySize() {{
  const canvas = document.getElementById("main-canvas");
  const stage = document.getElementById("canvas-stage");
  const zoomLbl = document.getElementById("zoom-label");

  canvas.classList.remove("pixelated");
  if (state.zoomMode === 'fit') {{
    const availW = Math.max(320, stage.clientWidth - 24);
    const availH = Math.max(420, window.innerHeight * 0.76);
    const scale = Math.min(availW / CONFIG.width, availH / CONFIG.height);
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    zoomLbl.textContent = `${{Math.round(scale * 100)}}%`;
    document.getElementById("zoom-slider").value = Math.min(300, Math.max(25, Math.round(scale * 100)));
  }} else if (state.zoomMode === 'fill') {{
    const availW = Math.max(320, stage.clientWidth - 24);
    const scale = availW / CONFIG.width;
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    zoomLbl.textContent = `${{Math.round(scale * 100)}}%`;
    document.getElementById("zoom-slider").value = Math.min(300, Math.max(25, Math.round(scale * 100)));
  }} else if (state.zoomMode === '1to1') {{
    canvas.style.width = `${{CONFIG.width}}px`;
    canvas.style.height = `${{CONFIG.height}}px`;
    canvas.classList.add("pixelated");
    zoomLbl.textContent = "100%";
    document.getElementById("zoom-slider").value = 100;
  }} else {{
    const scale = state.customZoom / 100.0;
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    if (scale >= 1.5) canvas.classList.add("pixelated");
    zoomLbl.textContent = `${{state.customZoom}}%`;
  }}
}}

function setStripMode(mode) {{
  state.stripMode = mode;
  document.getElementById("strip-mode-color").className = "btn" + (mode === 'color' ? " active" : "");
  document.getElementById("strip-mode-gray").className = "btn" + (mode === 'gray' ? " active" : "");
  requestRender();
}}

function drawScaleBarOnContext(ctx, width, height) {{
  const sb = state.scaleBar;
  if (!sb.enabled || sb.umPerPx <= 0 || sb.um <= 0) return;

  const rawBarPx = Math.round(sb.um / sb.umPerPx);
  if (rawBarPx < 2) return;

  const scaleMult = (sb.sizeScale || 100) / 100.0;
  const baseDim = Math.max(width, height);
  const marginX = Math.max(14, Math.round(width * 0.035));
  const marginY = Math.max(14, Math.round(height * 0.035));
  const barPx = Math.min(rawBarPx, Math.max(10, width - 2 * marginX));
  const barHeight = Math.max(4, Math.round(baseDim * 0.012 * scaleMult));
  const fontSize = Math.max(12, Math.round(baseDim * 0.025 * scaleMult));

  ctx.save();
  ctx.font = `bold ${{fontSize}}px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif`;
  const label = `${{Number.isInteger(sb.um) ? sb.um : sb.um}} μm`;
  const textWidth = ctx.measureText(label).width;
  const boxWidth = Math.max(barPx, textWidth);

  let x1, y1;
  if (sb.pos === 'bl') {{
    x1 = marginX;
    y1 = height - marginY - barHeight;
  }} else if (sb.pos === 'tr') {{
    x1 = width - marginX - boxWidth;
    y1 = marginY + fontSize + 6;
  }} else if (sb.pos === 'tl') {{
    x1 = marginX;
    y1 = marginY + fontSize + 6;
  }} else {{
    x1 = width - marginX - boxWidth;
    y1 = height - marginY - barHeight;
  }}

  if (sb.showBg !== false) {{
    const padX = Math.max(6, Math.round(fontSize * 0.4));
    const padY = Math.max(5, Math.round(fontSize * 0.35));
    const bgX = x1 - padX;
    const bgY = y1 - fontSize - padY - 4;
    const bgW = boxWidth + padX * 2;
    const bgH = fontSize + barHeight + padY * 2 + 4;

    ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
    ctx.fillRect(bgX, bgY, bgW, bgH);
  }}

  const barX = x1 + Math.round((boxWidth - barPx) / 2);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(barX, y1, barPx, barHeight);

  const textX = x1 + Math.round((boxWidth - textWidth) / 2);
  ctx.textBaseline = "bottom";
  ctx.fillText(label, textX, y1 - 3);
  ctx.restore();
}}

const mainCanvas = document.getElementById("main-canvas");
const mainCtx = mainCanvas.getContext("2d", {{ willReadFrequently: true }});
const mainImgData = mainCtx.createImageData(CONFIG.width, CONFIG.height);

function renderAll() {{
  const W = CONFIG.width;
  const H = CONFIG.height;
  const numPixels = W * H;
  const out = mainImgData.data;

  const activeCols = [];
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cfg = state.channels[c];
    const [r, g, b] = hexToRgbNorm(cfg.hex);
    activeCols.push({{
      enabled: cfg.enabled,
      r255: r * 255.0,
      g255: g * 255.0,
      b255: b * 255.0,
      lut: lutNorm[c],
      plane: visPlanesU16[c]
    }});
  }}

  for (let i = 0; i < numPixels; i++) {{
    let rSum = 0.0, gSum = 0.0, bSum = 0.0;
    for (let c = 0; c < CONFIG.numChannels; c++) {{
      const ac = activeCols[c];
      if (!ac.enabled) continue;
      const norm = ac.lut[ac.plane[i]];
      if (norm > 0.0) {{
        rSum += norm * ac.r255;
        gSum += norm * ac.g255;
        bSum += norm * ac.b255;
      }}
    }}
    const idx = i << 2;
    out[idx]     = rSum >= 255.0 ? 255 : (rSum + 0.5) | 0;
    out[idx + 1] = gSum >= 255.0 ? 255 : (gSum + 0.5) | 0;
    out[idx + 2] = bSum >= 255.0 ? 255 : (bSum + 0.5) | 0;
    out[idx + 3] = 255;
  }}

  mainCtx.putImageData(mainImgData, 0, 0);
  drawScaleBarOnContext(mainCtx, W, H);

  renderIndividualChannelCanvases(activeCols, numPixels, W, H);
  renderHistogram(activeCols, numPixels);
}}

function renderIndividualChannelCanvases(activeCols, numPixels, W, H) {{
  const isGray = state.stripMode === 'gray';
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cCanvas = document.getElementById(`thumb-canvas-${{c}}`);
    if (!cCanvas) continue;
    const ctx = cCanvas.getContext("2d");
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;
    const ac = activeCols[c];
    const lut = ac.lut;
    const plane = ac.plane;
    const rM = isGray ? 255.0 : ac.r255;
    const gM = isGray ? 255.0 : ac.g255;
    const bM = isGray ? 255.0 : ac.b255;

    for (let i = 0; i < numPixels; i++) {{
      const norm = lut[plane[i]];
      const idx = i << 2;
      data[idx]     = (norm * rM + 0.5) | 0;
      data[idx + 1] = (norm * gM + 0.5) | 0;
      data[idx + 2] = (norm * bM + 0.5) | 0;
      data[idx + 3] = 255;
    }}
    ctx.putImageData(imgData, 0, 0);
    drawScaleBarOnContext(ctx, W, H);
  }}
}}

function renderHistogram(activeCols, numPixels) {{
  const hCanvas = document.getElementById("hist-canvas");
  if (!hCanvas) return;
  const ctx = hCanvas.getContext("2d");
  const W = hCanvas.width;
  const H = hCanvas.height;

  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "#1f2630";
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(f => {{
    const x = Math.round(W * f);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }});

  const numBins = 128;
  const step = Math.max(1, Math.floor(numPixels / 16384));
  const bins = new Uint32Array(numBins);

  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const ac = activeCols[c];
    if (!ac.enabled) continue;
    bins.fill(0);
    const lut = ac.lut;
    const plane = ac.plane;
    for (let i = 0; i < numPixels; i += step) {{
      const norm = lut[plane[i]];
      const bIdx = Math.min(numBins - 1, (norm * numBins) | 0);
      bins[bIdx]++;
    }}

    let maxLog = 0.0;
    const logBins = new Float32Array(numBins);
    for (let b = 0; b < numBins; b++) {{
      const lv = Math.log1p(bins[b]);
      logBins[b] = lv;
      if (lv > maxLog) maxLog = lv;
    }}
    if (maxLog <= 0) continue;

    const hex = state.channels[c].hex;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let b = 0; b < numBins; b++) {{
      const x = (b / (numBins - 1)) * W;
      const y = H - (logBins[b] / maxLog) * (H - 10);
      ctx.lineTo(x, y);
    }}
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fillStyle = hex + "44";
    ctx.fill();

    ctx.strokeStyle = hex;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }}
}}

function getCleanExportPrefix() {{
  const raw = (document.getElementById("export-filename").value || CONFIG.groupPrefix || "microscopy").trim();
  return raw.replace(/[^a-zA-Z0-9._-]+/g, "_");
}}

function triggerBlobDownload(blob, filename) {{
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {{
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}, 250);
}}

function b64ToUint8Array(b64) {{
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}}

function encodeRawAnalysisMinisblackTiff(selectedIndices, width, height) {{
  const numChannelsSelected = selectedIndices.length;
  const numPixels = width * height;

  let targetBits = 8;
  for (const idx of selectedIndices) {{
    if (rawAnalysisBuffers[idx].bits > 8) targetBits = 16;
  }}

  const tmplKey = `${{targetBits}}_${{numChannelsSelected}}`;
  const tmpl = CONFIG.imagejTemplates[tmplKey];
  const prefixBytes = b64ToUint8Array(tmpl.prefixB64);
  const suffixBytes = b64ToUint8Array(tmpl.suffixB64);

  const blobParts = [prefixBytes];
  for (let p = 0; p < numChannelsSelected; p++) {{
    const chIdx = selectedIndices[p];
    const rawObj = rawAnalysisBuffers[chIdx];
    if (rawObj.bits === targetBits) {{
      blobParts.push(rawObj.byteView);
    }} else {{
      const promoted = new Uint16Array(numPixels);
      const src = rawObj.typedView;
      for (let i = 0; i < numPixels; i++) promoted[i] = src[i];
      blobParts.push(new Uint8Array(promoted.buffer));
    }}
  }}
  if (suffixBytes.length > 0) {{
    blobParts.push(suffixBytes);
  }}
  return new Blob(blobParts, {{ type: "image/tiff" }});
}}

function downloadRawAnalysisTIFF() {{
  const selected = [];
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    if (state.analysisSelected[c]) selected.push(c);
  }}
  if (selected.length === 0) {{
    alert("Please select at least one channel in '1A. Save for Analysis (Raw Data)'.");
    return;
  }}
  const chTag = selected.map(i => CONFIG.channels[i].keyword.replace(/\\s+/g, "")).join("-");
  const filename = `${{getCleanExportPrefix()}}_RAW_ANALYSIS_${{chTag}}_${{CONFIG.width}}x${{CONFIG.height}}.tif`;
  const blob = encodeRawAnalysisMinisblackTiff(selected, CONFIG.width, CONFIG.height);
  triggerBlobDownload(blob, filename);
}}

function downloadSingleChannelRawTIFF(c) {{
  const kw = CONFIG.channels[c].keyword.replace(/\\s+/g, "");
  const filename = `${{getCleanExportPrefix()}}_RAW_ANALYSIS_${{kw}}_${{CONFIG.width}}x${{CONFIG.height}}.tif`;
  const blob = encodeRawAnalysisMinisblackTiff([c], CONFIG.width, CONFIG.height);
  triggerBlobDownload(blob, filename);
}}

function encodePresentationRGB8Tiff(ctx, width, height) {{
  const imgData = ctx.getImageData(0, 0, width, height).data;
  const numPixels = width * height;
  const rgbByteLength = numPixels * 3;
  const headerSize = 8;
  const numTags = 12;
  const ifdSize = 2 + numTags * 12 + 4;
  const extraSize = 6 + 8 + 8;
  const totalSize = headerSize + rgbByteLength + (rgbByteLength % 2) + ifdSize + extraSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  view.setUint16(0, 0x4949, true);
  view.setUint16(2, 42, true);
  const ifdOffset = headerSize + rgbByteLength + (rgbByteLength % 2);
  view.setUint32(4, ifdOffset, true);

  let dst = headerSize;
  for (let i = 0; i < numPixels; i++) {{
    const src = i << 2;
    bytes[dst++] = imgData[src];
    bytes[dst++] = imgData[src + 1];
    bytes[dst++] = imgData[src + 2];
  }}

  let pos = ifdOffset;
  view.setUint16(pos, numTags, true);
  pos += 2;

  const extraOffsetStart = ifdOffset + ifdSize;
  const bpsOffset = extraOffsetStart;
  const xResOffset = extraOffsetStart + 6;
  const yResOffset = extraOffsetStart + 14;

  function writeTag(tag, type, count, valueOrOffset) {{
    view.setUint16(pos, tag, true);
    view.setUint16(pos + 2, type, true);
    view.setUint32(pos + 4, count, true);
    if (type === 3 && count === 1) {{
      view.setUint16(pos + 8, valueOrOffset, true);
      view.setUint16(pos + 10, 0, true);
    }} else {{
      view.setUint32(pos + 8, valueOrOffset, true);
    }}
    pos += 12;
  }}

  writeTag(256, 4, 1, width);
  writeTag(257, 4, 1, height);
  writeTag(258, 3, 3, bpsOffset);
  writeTag(259, 3, 1, 1);
  writeTag(262, 3, 1, 2);
  writeTag(273, 4, 1, headerSize);
  writeTag(277, 3, 1, 3);
  writeTag(278, 4, 1, height);
  writeTag(279, 4, 1, rgbByteLength);
  writeTag(282, 5, 1, xResOffset);
  writeTag(283, 5, 1, yResOffset);
  writeTag(296, 3, 1, 1);

  view.setUint32(pos, 0, true);

  view.setUint16(bpsOffset, 8, true);
  view.setUint16(bpsOffset + 2, 8, true);
  view.setUint16(bpsOffset + 4, 8, true);

  const pxPerUm = state.scaleBar.umPerPx > 0 ? (1.0 / state.scaleBar.umPerPx) : 1.0;
  const resNum = Math.max(1, Math.round(pxPerUm * 10000));
  view.setUint32(xResOffset, resNum, true);
  view.setUint32(xResOffset + 4, 10000, true);
  view.setUint32(yResOffset, resNum, true);
  view.setUint32(yResOffset + 4, 10000, true);

  return new Blob([buffer], {{ type: "image/tiff" }});
}}

function exportCanvasInPresentationFormat(canvas, ctx, baseName) {{
  const fmt = document.getElementById("pres-format-select").value || "PNG";
  if (fmt === "TIFF") {{
    const filename = `${{baseName}}_8bit_RGB.tif`;
    const blob = encodePresentationRGB8Tiff(ctx, CONFIG.width, CONFIG.height);
    triggerBlobDownload(blob, filename);
  }} else if (fmt === "JPEG") {{
    const filename = `${{baseName}}.jpg`;
    canvas.toBlob(blob => {{
      if (blob) triggerBlobDownload(blob, filename);
    }}, "image/jpeg", 0.96);
  }} else {{
    const filename = `${{baseName}}.png`;
    canvas.toBlob(blob => {{
      if (blob) triggerBlobDownload(blob, filename);
    }}, "image/png");
  }}
}}

function downloadPresentationImage() {{
  const baseName = `${{getCleanExportPrefix()}}_PRESENTATION_${{CONFIG.width}}x${{CONFIG.height}}`;
  exportCanvasInPresentationFormat(mainCanvas, mainCtx, baseName);
}}

function downloadSingleChannelPresentation(c) {{
  const cCanvas = document.getElementById(`thumb-canvas-${{c}}`);
  if (!cCanvas) return;
  const ctx = cCanvas.getContext("2d");
  const kw = CONFIG.channels[c].keyword.replace(/\\s+/g, "");
  const baseName = `${{getCleanExportPrefix()}}_PRESENTATION_${{kw}}_${{CONFIG.width}}x${{CONFIG.height}}`;
  exportCanvasInPresentationFormat(cCanvas, ctx, baseName);
}}

function downloadAllSelectedChannelsPresentation() {{
  downloadPresentationImage();
  let delay = 250;
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    if (state.channels[c].enabled) {{
      setTimeout(() => downloadSingleChannelPresentation(c), delay);
      delay += 250;
    }}
  }}
}}

initStudioState();
for (let c = 0; c < CONFIG.numChannels; c++) {{
  updateChannelLUT(c);
}}
buildUI();
updateUndoRedoButtons();
applyCanvasDisplaySize();
renderAll();
window.addEventListener("resize", () => {{
  if (state.zoomMode === 'fit' || state.zoomMode === 'fill') applyCanvasDisplaySize();
}});
window.addEventListener("pointerup", commitContinuousEdit);
window.addEventListener("mouseup", commitContinuousEdit);
window.addEventListener("change", commitContinuousEdit);
window.addEventListener("keydown", handleGlobalKeyDown);
try {{
  if (window.parent && window.parent.document) {{
    window.parent.document.addEventListener("keydown", handleGlobalKeyDown);
  }}
}} catch (e) {{}}
</script>
</body>
</html>
"""
    return html
