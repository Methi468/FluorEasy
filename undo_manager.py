"""
Session State History & Undo/Redo Manager for the Legacy Microscope Batch Processor.

Features:
- Tracks all UI tweaks across scalebar parameters (Objective Lens, Scalebar Length μm,
  Calibration Factor μm/pixel, position, color, visibility), per-channel pseudocolor
  rendering adjustments (Brightness, Contrast, Gamma, Weight, Auto-Contrast), and
  per-group channel selection checkboxes.
- Supports multi-step Undo and Redo via Streamlit `on_click` callbacks (avoiding widget key lock issues).
- Injects a zero-height HTML/JS keyboard shortcut listener on `window.parent.document`
  that binds `Ctrl+Z` (and `Cmd+Z` on macOS) to trigger the Undo button, and `Ctrl+Y` / `Ctrl+Shift+Z`
  to trigger the Redo button.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from batch_grouping import ALLOWED_CHANNELS


# Default tracked settings
DEFAULT_TRACKED_STATE: Dict[str, Any] = {
    # Scalebar Panel settings
    "sb_enabled": True,
    "sb_objective": "40x",
    "sb_length_um": 50.0,
    "sb_calibration": 0.1625,
    "sb_position": "Bottom Right",
    "sb_color": "White",
    "sb_backing_box": True,
    "sb_show_obj_label": True,
    # Per-channel rendering tweaks
    **{f"ch_brightness_{ch}": 0.0 for ch in ALLOWED_CHANNELS},
    **{f"ch_contrast_{ch}": 0.0 for ch in ALLOWED_CHANNELS},
    **{f"ch_gamma_{ch}": 1.0 for ch in ALLOWED_CHANNELS},
    **{f"ch_weight_{ch}": 1.0 for ch in ALLOWED_CHANNELS},
    **{f"ch_auto_{ch}": False for ch in ALLOWED_CHANNELS},
}

FRIENDLY_KEY_LABELS: Dict[str, str] = {
    "sb_enabled": "Scalebar Enabled",
    "sb_objective": "Objective Lens",
    "sb_length_um": "Scalebar Length (μm)",
    "sb_calibration": "Calibration Factor (μm/pixel)",
    "sb_position": "Scalebar Position",
    "sb_color": "Scalebar Color",
    "sb_backing_box": "Scalebar Backing Box",
    "sb_show_obj_label": "Show Objective on Scalebar",
}
for _ch in ALLOWED_CHANNELS:
    _ch_title = _ch.title()
    FRIENDLY_KEY_LABELS[f"ch_brightness_{_ch}"] = f"{_ch_title} Brightness"
    FRIENDLY_KEY_LABELS[f"ch_contrast_{_ch}"] = f"{_ch_title} Contrast"
    FRIENDLY_KEY_LABELS[f"ch_gamma_{_ch}"] = f"{_ch_title} Gamma"
    FRIENDLY_KEY_LABELS[f"ch_weight_{_ch}"] = f"{_ch_title} Blend Weight"
    FRIENDLY_KEY_LABELS[f"ch_auto_{_ch}"] = f"{_ch_title} Auto-Contrast"


@dataclass
class HistoryEntry:
    """Represents a single snapshot on the undo/redo stack."""
    state: Dict[str, Any]
    description: str


def _get_all_tracked_keys() -> List[str]:
    """Return all static and dynamic (per-group) tracked session state keys."""
    keys = list(DEFAULT_TRACKED_STATE.keys())
    for k in list(st.session_state.keys()):
        if k.startswith("grp_sel_") or k.startswith("grp_fmt_"):
            keys.append(k)
    return sorted(set(keys))


def init_undo_state() -> None:
    """Initialize default session state values and undo/redo stacks if not present."""
    for key, default_val in DEFAULT_TRACKED_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    if "_undo_stack" not in st.session_state:
        st.session_state["_undo_stack"] = []

    if "_redo_stack" not in st.session_state:
        st.session_state["_redo_stack"] = []

    if "_committed_state" not in st.session_state:
        st.session_state["_committed_state"] = {
            k: copy.deepcopy(st.session_state[k])
            for k in _get_all_tracked_keys()
            if k in st.session_state
        }

    if "_last_undo_notice" not in st.session_state:
        st.session_state["_last_undo_notice"] = None


def _format_val(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def _describe_diff(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Optional[str]:
    """Create a concise human-readable description of what changed between two states."""
    changes: List[str] = []
    for k, new_val in new_state.items():
        if k not in old_state:
            continue
        old_val = old_state[k]
        if old_val != new_val:
            if k in FRIENDLY_KEY_LABELS:
                label = FRIENDLY_KEY_LABELS[k]
            elif k.startswith("grp_sel_"):
                parts = k[len("grp_sel_"):].rsplit("_", 1)
                if len(parts) == 2:
                    label = f"Group '{parts[0]}' [{parts[1].title()}]"
                else:
                    label = k
            elif k.startswith("grp_fmt_"):
                grp_name = k[len("grp_fmt_"):]
                label = f"Group '{grp_name}' Format"
            else:
                label = k
            changes.append(f"{label}: {_format_val(old_val)} → {_format_val(new_val)}")

    if not changes:
        return None
    if len(changes) == 1:
        return changes[0]
    return f"{changes[0]} (+{len(changes) - 1} more)"


def sync_state_history() -> None:
    """
    Compare current tracked session_state keys against `_committed_state`.
    If the user modified any UI control since the last run, push the prior committed
    state onto `_undo_stack` and update `_committed_state`.
    """
    init_undo_state()

    committed: Dict[str, Any] = st.session_state["_committed_state"]
    tracked_keys = _get_all_tracked_keys()

    current_snapshot: Dict[str, Any] = {
        k: copy.deepcopy(st.session_state[k])
        for k in tracked_keys
        if k in st.session_state
    }

    # Add any newly registered dynamic keys (e.g. newly uploaded group checkboxes) to committed
    for k, v in current_snapshot.items():
        if k not in committed:
            committed[k] = copy.deepcopy(v)

    diff_desc = _describe_diff(committed, current_snapshot)
    if diff_desc is not None:
        undo_stack: List[HistoryEntry] = st.session_state["_undo_stack"]
        undo_stack.append(
            HistoryEntry(
                state=copy.deepcopy(committed),
                description=diff_desc,
            )
        )
        # Cap history length at 100 entries
        if len(undo_stack) > 100:
            del undo_stack[0]

        st.session_state["_redo_stack"] = []
        st.session_state["_committed_state"] = copy.deepcopy(current_snapshot)


def perform_undo() -> None:
    """
    Callback executed when the user clicks 'Undo' or presses Ctrl+Z.
    Restores the most recent snapshot from `_undo_stack`.
    """
    undo_stack: List[HistoryEntry] = st.session_state.get("_undo_stack", [])
    if not undo_stack:
        return

    entry = undo_stack.pop()
    current_committed = copy.deepcopy(st.session_state.get("_committed_state", {}))
    st.session_state.setdefault("_redo_stack", []).append(
        HistoryEntry(state=current_committed, description=entry.description)
    )

    for k, val in entry.state.items():
        st.session_state[k] = copy.deepcopy(val)

    st.session_state["_committed_state"] = copy.deepcopy(entry.state)
    st.session_state["_last_undo_notice"] = f"↩️ Undid: {entry.description}"


def perform_redo() -> None:
    """
    Callback executed when the user clicks 'Redo' or presses Ctrl+Y / Ctrl+Shift+Z.
    """
    redo_stack: List[HistoryEntry] = st.session_state.get("_redo_stack", [])
    if not redo_stack:
        return

    entry = redo_stack.pop()
    current_committed = copy.deepcopy(st.session_state.get("_committed_state", {}))
    st.session_state.setdefault("_undo_stack", []).append(
        HistoryEntry(state=current_committed, description=entry.description)
    )

    for k, val in entry.state.items():
        st.session_state[k] = copy.deepcopy(val)

    st.session_state["_committed_state"] = copy.deepcopy(entry.state)
    st.session_state["_last_undo_notice"] = f"↪️ Redid: {entry.description}"


def perform_reset_defaults() -> None:
    """
    Callback to reset scalebar and channel adjustments to default values (itself undoable).
    """
    current_committed = copy.deepcopy(st.session_state.get("_committed_state", {}))
    st.session_state.setdefault("_undo_stack", []).append(
        HistoryEntry(state=current_committed, description="Reset all settings to defaults")
    )
    st.session_state["_redo_stack"] = []

    for k, default_val in DEFAULT_TRACKED_STATE.items():
        st.session_state[k] = copy.deepcopy(default_val)

    new_committed = copy.deepcopy(current_committed)
    for k, default_val in DEFAULT_TRACKED_STATE.items():
        new_committed[k] = copy.deepcopy(default_val)

    st.session_state["_committed_state"] = new_committed
    st.session_state["_last_undo_notice"] = "🔄 Reset all adjustments to default values (press Undo to restore)"


def inject_ctrl_z_shortcut() -> None:
    """
    Inject an invisible HTML/JS component that binds Ctrl+Z / Cmd+Z on the parent
    Streamlit document to click the prominent 'Undo (Ctrl+Z)' button, and Ctrl+Y / Cmd+Shift+Z
    to click 'Redo'.
    """
    shortcut_html = """
    <script>
    (function() {
        try {
            const parentDoc = window.parent.document;
            if (!parentDoc) return;
            if (parentDoc._legacyMicroscopeUndoListenerAttached) return;
            parentDoc._legacyMicroscopeUndoListenerAttached = true;

            parentDoc.addEventListener('keydown', function(e) {
                const isMod = e.ctrlKey || e.metaKey;
                if (!isMod) return;

                const keyLower = (e.key || '').toLowerCase();
                const isUndo = !e.shiftKey && keyLower === 'z';
                const isRedo = (e.shiftKey && keyLower === 'z') || (!e.shiftKey && keyLower === 'y');

                if (isUndo) {
                    const hiddenBtn = parentDoc.querySelector('.st-key-hidden_undo_trigger button');
                    if (hiddenBtn && !hiddenBtn.disabled) {
                        e.preventDefault();
                        e.stopPropagation();
                        hiddenBtn.click();
                        return;
                    }
                } else if (isRedo) {
                    const hiddenRedo = parentDoc.querySelector('.st-key-hidden_redo_trigger button');
                    if (hiddenRedo && !hiddenRedo.disabled) {
                        e.preventDefault();
                        e.stopPropagation();
                        hiddenRedo.click();
                        return;
                    }
                }
            }, true);
        } catch (err) {
            console.warn('Could not bind Ctrl+Z shortcut to parent document:', err);
        }
    })();
    </script>
    """
    components.html(shortcut_html, height=0, width=0)
