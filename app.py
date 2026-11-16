"""
FluorEasy — Batch Microscope Image Viewer & Real-Time Studio.

Matches the dark-themed Real-Time HTML5/Canvas Studio UI with instant 60 FPS
client-side rendering while fulfilling all 5 core specifications:
1. Smart Batch Upload & Auto-Grouping (Left Sidebar):
   - Upload tens or hundreds of grayscale TIFF or JPEG files at once.
   - Automatically groups files by their exact filename prefix.
   - Determines channel/color strictly from the last word of the filename (excluding extension):
     allowed keywords (case-insensitive): 'bright field', 'blue', 'green', 'red'.
   - Immediately displays `st.error` if a file's last word is not one of these four,
     instructing the user to rename the file with the correct color at the end.
   - Groups up to 4 files per prefix (1–4 channels), and shows `st.error` / splits prefixes with >4 files.
2. Image Rendering & Pseudocoloring (Main Studio + Right Dock):
   - Applies DAPI style to 'blue', FITC style to 'green', TRITC style to 'red',
     and Grayscale/Phase contrast to 'bright field'.
   - Displays the merged composite and individual channels at 60 FPS in the Real-Time Studio.
3. Scalebar Panel (Right Studio Dock):
   - Dropdown & quick buttons to select Objective Lens: 4x, 10x, 16x, 20x, 40x, 100x.
   - Input field for Scalebar Length in micrometers (μm).
   - Input field for 'Calibration Factor (μm/pixel)' with a clear UI explanation that this is
     hardware-specific to their camera sensor and cannot be pulled from generic online databases.
4. Export / Save Options Per Group (Right Studio Dock):
   - Channel selection checkboxes to choose which channels to include in exports.
   - 'Save for Analysis': Exports selected channels as a raw, multi-dimensional TIFF using `tifffile`
     templates preserving original grayscale intensity data perfectly (no LUTs, no flattening).
   - 'Save for Presentation': Exports the visually pseudocolored and merged image with the scalebar
     burned in (PNG, JPEG, or 8-bit RGB TIFF).
5. Undo Functionality:
   - Full client-side and session-state undo/redo stack bound to Ctrl+Z (and Cmd+Z).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from batch_grouping import (
    ImageGroup,
    group_uploaded_files,
)
from image_processing import (
    ScalebarConfig,
    load_raw_grayscale_array,
)
from realtime_studio import build_legacy_studio_html


st.set_page_config(
    page_title="FluorEasy — Batch Microscope Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Remove sticky top white header bar so the dark studio sits flush at the top */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto;
    }
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton,
    [data-testid="stMainMenu"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=512)
def cached_decode_raw_grayscale(file_bytes: bytes, filename: str) -> np.ndarray:
    """Cache decoded immutable 2D raw grayscale arrays (`uint8` or `uint16`)."""
    return load_raw_grayscale_array(file_bytes, filename)


def _extract_bytes(file_obj: Any) -> bytes:
    """Extract raw bytes from an UploadedFile or bytes object without copying unnecessarily."""
    if isinstance(file_obj, (bytes, bytearray)):
        return bytes(file_obj)
    file_obj.seek(0)
    data = file_obj.read()
    file_obj.seek(0)
    return data


def _decode_single_group(grp: ImageGroup) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Decode only the channels belonging to `grp` (up to 4 files) for instant responsiveness."""
    ch_arrays: Dict[str, np.ndarray] = {}
    errors: List[str] = []
    for ch, item in grp.files_by_channel.items():
        try:
            raw_bytes = _extract_bytes(item.file_obj)
            ch_arrays[ch] = cached_decode_raw_grayscale(raw_bytes, item.filename)
        except Exception as exc:
            errors.append(f"Failed to decode `{item.filename}`: {exc}")
    return ch_arrays, errors


def main() -> None:
    # --- LEFT SIDEBAR: 1. DATA SOURCE (SMART BATCH UPLOAD) ---
    with st.sidebar:
        st.markdown("## 🔬 FluorEasy")
        st.caption("Batch Legacy Microscope Viewer & Real-Time Studio")
        st.header("📂 1. Data Source")
        uploaded_files = st.file_uploader(
            "Batch Upload Grayscale TIFF / JPEG",
            type=["tif", "tiff", "jpg", "jpeg"],
            accept_multiple_files=True,
            help=(
                "Upload tens or hundreds of grayscale .tif/.tiff or .jpg/.jpeg files at once. "
                "Files are grouped by prefix, and channel is determined strictly by the last word "
                "before the extension: 'bright field', 'blue', 'green', or 'red'."
            ),
        )

    test_items = st.session_state.get("_test_uploaded_items")
    if not uploaded_files and not test_items:
        st.info(
            "👈 Upload your legacy grayscale **TIFF (`.tif`, `.tiff`)** or **JPEG (`.jpg`, `.jpeg`)** "
            "files in the sidebar to begin."
        )
        return

    # Collect active batch items
    if uploaded_files:
        active_items: List[Tuple[str, Any]] = [(uf.name, uf) for uf in uploaded_files]
    else:
        active_items = list(test_items)

    # Run Smart Auto-Grouping
    grouping_result = group_uploaded_files(active_items, split_overflow=True)

    # Immediately show st.error if any file's last word is not 'bright field', 'blue', 'green', or 'red'
    if grouping_result.invalid_files:
        for bad_item in grouping_result.invalid_files:
            st.error(f"🚨 **Filename Error:** {bad_item.error_message}")

    # Show st.error if any prefix has >4 files or duplicate channel keywords
    if grouping_result.overflow_errors:
        for overflow_msg in grouping_result.overflow_errors:
            st.error(f"⚠️ **Prefix Group Size Error:** {overflow_msg}")

    if not grouping_result.groups:
        st.warning(
            "No valid image groups were found. Please rename the flagged files so their last word "
            "before the extension is `bright field`, `blue`, `green`, or `red`."
        )
        return

    # --- LEFT SIDEBAR: 2. GROUP SELECTION (AUTO-GROUPED BY PREFIX) ---
    with st.sidebar:
        st.divider()
        st.header("🧬 2. Image Groups")
        st.caption(
            f"✅ **{len(grouping_result.groups)} groups** auto-grouped from **{grouping_result.total_uploaded} files**"
        )

        group_indices = list(range(len(grouping_result.groups)))
        selected_grp_idx = st.selectbox(
            "Select Group (Filename Prefix)",
            options=group_indices,
            key="selected_group_idx",
            format_func=lambda idx: (
                f"Group {idx + 1}: {grouping_result.groups[idx].group_id} "
                f"({grouping_result.groups[idx].file_count} ch: "
                f"{', '.join(grouping_result.groups[idx].available_channels)})"
            ),
        )
        active_group = grouping_result.groups[selected_grp_idx]

        with st.expander("📋 Batch Grouping Table", expanded=False):
            rows = [
                {
                    "Prefix": g.group_id,
                    "Ch": g.file_count,
                    "Channels": ", ".join(g.available_channels),
                }
                for g in grouping_result.groups
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Decode ONLY the currently selected group (1–4 files) -> ultra-fast!
    raw_map, decode_errors = _decode_single_group(active_group)
    for derr in decode_errors:
        st.error(derr)

    if not raw_map:
        st.error(f"Could not decode any channels for group `{active_group.group_id}`.")
        return

    # --- MAIN VIEWPORT: FULL-SPEED REAL-TIME HTML5 STUDIO (FluorEasy UI) ---
    studio_html = build_legacy_studio_html(
        group=active_group,
        raw_channels=raw_map,
        scalebar_cfg=ScalebarConfig(),
        initial_selected_channels=list(active_group.available_channels),
    )
    components.html(studio_html, height=1220, scrolling=True)


if __name__ == "__main__":
    main()
