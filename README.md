# 🔬 FluorEasy — Batch Microscope Image Studio

A high-speed Streamlit & HTML5 Canvas studio for batch processing native grayscale legacy microscope images (`.tif`, `.tiff`, `.jpg`, `.jpeg`).

## Key Features

1. **Smart Batch Upload & Auto-Grouping**
   - Upload tens or hundreds of grayscale TIFF or JPEG files at once.
   - Automatically groups files by their **filename prefix** and determines the channel/color strictly from the **last word** of the filename before the extension (`bright field`, `blue`, `green`, `red` — case-insensitive).
   - Displays an immediate error if a file's last word is not one of the four allowed keywords, instructing the user how to rename the file.
   - Groups up to 4 files per prefix (one per channel), handling 1–4 files per prefix and flagging/splitting prefixes with >4 files.

2. **Real-Time 60 FPS Image Rendering & Pseudocoloring**
   - Maps each grayscale channel to its default fluorophore LUT:
     - `blue` → **Alexa 405 / DAPI (Blue)** (`#3B82F6` — DAPI / Hoechst)
     - `green` → **Alexa 488 / GFP (Green)** (`#00FF44`)
     - `red` → **Alexa 594 / mCherry (Red)** (`#FF1A1A`)
     - `bright field` → **Grayscale / White** (`#FFFFFF`)
   - Includes a full fluorophore LUT preset menu (`Alexa 405`, `488`, `532`, `546`, `568`, `594`, `647`, `680`, `Cyan`, `Yellow`, `Grayscale`, `Custom Color`) and real-time Brightness, Contrast, Exposure (EV), Gamma ($\gamma$), and Black/White Point sliders.

3. **Calibrated Scalebar Panel**
   - Select **Objective Lens**: `4x`, `10x`, `16x`, `20x`, `40x`, `100x`.
   - Define **Scalebar Length (μm)** and hardware-specific **Calibration Factor (μm/pixel)**.
   - Customize corner placement, text/bar thickness, and toggle the **Scalebar Background Box** on or off.

4. **Dual Export / Save Pipelines (Per Group)**
   - **1A. Save for Analysis (Raw Data)**: Select channels to export as an unmodified, raw, multi-dimensional TIFF (`axes='CYX'`) via `tifffile`, preserving native grayscale integer intensities (`uint8` / `uint16`) with zero LUTs and zero flattening.
   - **1B. Save for Presentation (Visual Render)**: Export the visually pseudocolored and merged composite with the calibrated scalebar burned in (`PNG`, `JPEG`, or `8-bit RGB TIFF`).

5. **Undo / Redo (`Ctrl+Z` / `Cmd+Z`)**
   - Multi-step undo/redo history stack bound to `Ctrl+Z` (or `Cmd+Z` on macOS) and `Ctrl+Y` / `Cmd+Shift+Z`.

## Installation & Quick Start

```bash
git clone <your-repo-url>
cd FluorEasy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS, you can also double-click **`FluorEasy.command`** in Finder to launch the app directly.
