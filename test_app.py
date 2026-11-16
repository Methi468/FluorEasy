"""
Comprehensive unit and integration test suite for FluorEasy.
"""

import io
import os
import unittest
import numpy as np
from PIL import Image
import tifffile

from batch_grouping import (
    ALLOWED_CHANNELS,
    group_uploaded_files,
    parse_microscope_filename,
)
from image_processing import (
    OBJECTIVE_LENSES,
    ChannelRenderConfig,
    ScalebarConfig,
    draw_scalebar_overlay,
    export_group_for_analysis,
    export_group_for_presentation,
    load_raw_grayscale_array,
    render_group_composite,
)
from sample_dataset import generate_sample_batch_files
from realtime_studio import build_legacy_studio_html


class TestSmartBatchGrouping(unittest.TestCase):
    def test_valid_color_keywords_case_insensitive(self):
        cases = [
            ("Exp01_SampleA_blue.tif", "Exp01_SampleA", "blue"),
            ("Exp01_SampleA_GREEN.tiff", "Exp01_SampleA", "green"),
            ("Exp01_SampleA_Red.jpg", "Exp01_SampleA", "red"),
            ("Exp01_SampleA_bright field.tif", "Exp01_SampleA", "bright field"),
            ("Exp01_SampleA_BRIGHT_FIELD.jpeg", "Exp01_SampleA", "bright field"),
            ("Exp01_SampleA-bright-field.tif", "Exp01_SampleA", "bright field"),
            ("Slide 42 blue.jpg", "Slide 42", "blue"),
        ]
        for fname, expected_prefix, expected_ch in cases:
            parsed = parse_microscope_filename(fname)
            self.assertTrue(parsed.is_valid, f"Expected {fname} to be valid")
            self.assertEqual(parsed.prefix, expected_prefix)
            self.assertEqual(parsed.channel, expected_ch)
            self.assertIsNone(parsed.error_message)

    def test_invalid_last_word_triggers_error(self):
        bad_names = [
            "Exp01_SampleA_yellow.tif",
            "Exp01_SampleA_dapi.tif",
            "Exp01_SampleA_ch1.jpg",
            "Exp01_SampleA_blue_overlay.tif",
            "Exp01_SampleA_phase.tif",
        ]
        for fname in bad_names:
            parsed = parse_microscope_filename(fname)
            self.assertFalse(parsed.is_valid, f"Expected {fname} to be invalid")
            self.assertIsNotNone(parsed.error_message)
            self.assertIn("rename the file", parsed.error_message.lower())

    def test_grouping_2_3_4_and_100_files(self):
        # 2 files for GroupA, 3 files for GroupB, 4 files for GroupC
        items = [
            ("GroupA_blue.tif", b"1"),
            ("GroupA_green.tif", b"2"),
            ("GroupB_blue.tif", b"3"),
            ("GroupB_green.tif", b"4"),
            ("GroupB_red.tif", b"5"),
            ("GroupC_bright field.tif", b"6"),
            ("GroupC_blue.tif", b"7"),
            ("GroupC_green.tif", b"8"),
            ("GroupC_red.tif", b"9"),
        ]
        res = group_uploaded_files(items)
        self.assertEqual(len(res.invalid_files), 0)
        self.assertEqual(len(res.overflow_errors), 0)
        self.assertEqual(len(res.groups), 3)

        grp_map = {g.prefix: g for g in res.groups}
        self.assertEqual(grp_map["GroupA"].file_count, 2)
        self.assertEqual(grp_map["GroupB"].file_count, 3)
        self.assertEqual(grp_map["GroupC"].file_count, 4)

        # Test batch of 120 files (30 prefixes x 4 channels)
        large_batch = []
        for i in range(30):
            for ch in ALLOWED_CHANNELS:
                large_batch.append((f"BatchSlide_{i:03d}_{ch}.tif", b"dummy"))
        large_res = group_uploaded_files(large_batch)
        self.assertEqual(len(large_res.groups), 30)
        self.assertEqual(len(large_res.invalid_files), 0)

    def test_overflow_more_than_4_files_per_prefix(self):
        items = [
            ("OverflowPrefix_bright field.tif", b"1"),
            ("OverflowPrefix_blue.tif", b"2"),
            ("OverflowPrefix_green.tif", b"3"),
            ("OverflowPrefix_red.tif", b"4"),
            ("OverflowPrefix_BRIGHT_FIELD.tiff", b"5"),
        ]
        res = group_uploaded_files(items, split_overflow=True)
        self.assertEqual(len(res.overflow_errors), 1)
        self.assertIn("OverflowPrefix", res.overflow_errors[0])
        self.assertEqual(len(res.groups), 2)
        self.assertEqual(res.groups[0].file_count, 4)
        self.assertEqual(res.groups[1].file_count, 1)


class TestRenderingScalebarAndExports(unittest.TestCase):
    def setUp(self):
        # Create deterministic 16-bit raw grayscale test planes (64x80)
        self.raw_blue = np.full((64, 80), 32768, dtype=np.uint16)
        self.raw_green = np.full((64, 80), 16384, dtype=np.uint16)
        self.raw_red = np.full((64, 80), 65535, dtype=np.uint16)
        self.raw_bf = np.full((64, 80), 8192, dtype=np.uint16)
        self.raw_channels = {
            "blue": self.raw_blue,
            "green": self.raw_green,
            "red": self.raw_red,
            "bright field": self.raw_bf,
        }

    def test_pseudocoloring_luts(self):
        # Blue only -> only B channel (>0), R=0, G=0
        comp_blue = render_group_composite({"blue": self.raw_blue}, ["blue"])
        self.assertTrue(np.all(comp_blue[..., 0] == 0))
        self.assertTrue(np.all(comp_blue[..., 1] == 0))
        self.assertTrue(np.all(comp_blue[..., 2] > 120))

        # Green only -> only G channel (>0), R=0, B=0
        comp_green = render_group_composite({"green": self.raw_green}, ["green"])
        self.assertTrue(np.all(comp_green[..., 0] == 0))
        self.assertTrue(np.all(comp_green[..., 1] > 60))
        self.assertTrue(np.all(comp_green[..., 2] == 0))

        # Red only -> only R channel (>0), G=0, B=0
        comp_red = render_group_composite({"red": self.raw_red}, ["red"])
        self.assertTrue(np.all(comp_red[..., 0] == 255))
        self.assertTrue(np.all(comp_red[..., 1] == 0))
        self.assertTrue(np.all(comp_red[..., 2] == 0))

        # Bright field only -> R == G == B (grayscale)
        comp_bf = render_group_composite({"bright field": self.raw_bf}, ["bright field"])
        self.assertTrue(np.array_equal(comp_bf[..., 0], comp_bf[..., 1]))
        self.assertTrue(np.array_equal(comp_bf[..., 1], comp_bf[..., 2]))

    def test_scalebar_objectives_and_calibration(self):
        expected_lenses = ("4x", "10x", "16x", "20x", "40x", "100x")
        self.assertEqual(OBJECTIVE_LENSES, expected_lenses)

        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        cfg = ScalebarConfig(
            enabled=True,
            objective_lens="40x",
            length_um=25.0,
            calibration_um_per_px=0.25,  # 25 / 0.25 = 100 px
        )
        out_img, info = draw_scalebar_overlay(canvas, cfg)
        self.assertTrue(info["drawn"])
        self.assertEqual(info["bar_px"], 100)
        self.assertEqual(info["effective_bar_px"], 100)
        self.assertGreater(int(np.sum(out_img)), 0)

    def test_save_for_analysis_preserves_exact_raw_data(self):
        selected = ["blue", "green", "red"]
        tiff_bytes = export_group_for_analysis(
            raw_channels=self.raw_channels,
            selected_channels=selected,
            calibration_um_per_px=0.1625,
            objective_lens="40x",
        )
        reloaded = tifffile.imread(io.BytesIO(tiff_bytes))
        self.assertEqual(reloaded.shape, (3, 64, 80))
        self.assertEqual(reloaded.dtype, np.uint16)
        self.assertTrue(np.array_equal(reloaded[0], self.raw_blue))
        self.assertTrue(np.array_equal(reloaded[1], self.raw_green))
        self.assertTrue(np.array_equal(reloaded[2], self.raw_red))

    def test_save_for_presentation_formats(self):
        sb_cfg = ScalebarConfig(
            enabled=True,
            objective_lens="20x",
            length_um=20.0,
            calibration_um_per_px=0.5,
        )
        for fmt, expected_ext in [("PNG", "png"), ("JPEG", "jpg"), ("8-bit RGB TIFF", "tif")]:
            data, mime, ext = export_group_for_presentation(
                raw_channels=self.raw_channels,
                selected_channels=["bright field", "blue", "green", "red"],
                output_format=fmt,
                scalebar_cfg=sb_cfg,
            )
            self.assertEqual(ext, expected_ext)
            self.assertGreater(len(data), 100)
            if ext == "tif":
                arr = tifffile.imread(io.BytesIO(data))
            else:
                arr = np.asarray(Image.open(io.BytesIO(data)))
            self.assertEqual(arr.shape, (64, 80, 3))
            self.assertEqual(arr.dtype, np.uint8)

    def test_sample_batch_end_to_end(self):
        sample_files = generate_sample_batch_files()
        self.assertEqual(len(sample_files), 9)
        grouped = group_uploaded_files(sample_files)
        self.assertEqual(len(grouped.groups), 3)
        self.assertEqual(len(grouped.invalid_files), 0)

        for grp in grouped.groups:
            raw_map = {}
            for ch, item in grp.files_by_channel.items():
                arr = load_raw_grayscale_array(item.file_obj, item.filename)
                self.assertEqual(arr.ndim, 2)
                self.assertFalse(arr.flags.writeable)
                raw_map[ch] = arr
            studio_html = build_legacy_studio_html(
                group=grp,
                raw_channels=raw_map,
                scalebar_cfg=ScalebarConfig(),
            )
            self.assertIn("1A. Save for Analysis (Raw Data)", studio_html)
            self.assertIn("1B. Save for Presentation (Visual Render)", studio_html)
            self.assertIn("3. Scalebar Panel", studio_html)


class TestStreamlitUIAndUndo(unittest.TestCase):
    def test_streamlit_app_and_grouping_sidebar(self):
        from streamlit.testing.v1 import AppTest

        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        at = AppTest.from_file(
            app_path,
            default_timeout=30,
        )
        at.session_state["_test_uploaded_items"] = generate_sample_batch_files()
        at.run()
        self.assertEqual(len(at.exception), 0, f"Unexpected exceptions: {at.exception}")

        # Verify that the group selector in the left sidebar has all 3 groups
        grp_sel = at.selectbox(key="selected_group_idx")
        self.assertEqual(len(grp_sel.options), 3)
        self.assertEqual(grp_sel.value, 0)

        # Switch to Group 2 and verify clean execution
        grp_sel.select(1).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["selected_group_idx"], 1)


if __name__ == "__main__":
    unittest.main()
