"""
Smart Batch Upload & Auto-Grouping logic for legacy microscope images.

Specifications implemented:
- Strictly determines channel/color from the last word of the filename (excluding extension).
- Allowed color keywords (case-insensitive): 'bright field', 'blue', 'green', 'red'.
- Produces explicit error messages when a file's last word is not one of the four allowed colors,
  instructing the user to rename the file with the correct color at the end.
- Groups files sharing the exact same prefix into a group of up to 4 files (one per allowed channel).
- Handles 1, 2, 3, or 4 files per prefix naturally, and flags/splits prefixes with >4 files
  or duplicate channel assignments.
"""

from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# Canonical channel order
ALLOWED_CHANNELS: Tuple[str, ...] = ("bright field", "blue", "green", "red")

CHANNEL_DISPLAY_NAMES: Dict[str, str] = {
    "bright field": "Bright Field (Phase / Grayscale)",
    "blue": "Blue (DAPI Style)",
    "green": "Green (FITC Style)",
    "red": "Red (TRITC Style)",
}

CHANNEL_BADGE_COLORS: Dict[str, str] = {
    "bright field": "#94A3B8",
    "blue": "#3B82F6",
    "green": "#22C55E",
    "red": "#EF4444",
}

# Matches trailing 'bright field' (allowing space, underscore, or hyphen between bright & field)
_BRIGHT_FIELD_PATTERN = re.compile(
    r"^(.*?)(?:[\s_\-\.]+|^)(bright[\s_\-]*field)$",
    re.IGNORECASE,
)

# Matches the last word separated by space, underscore, hyphen, or dot
_LAST_WORD_PATTERN = re.compile(
    r"^(.*?)(?:[\s_\-\.]+|^)([^\s_\-\.]+)$"
)


@dataclass
class ParsedFileItem:
    """Represents a single uploaded file after filename parsing."""
    filename: str
    file_obj: Any
    is_valid: bool
    prefix: str
    channel: Optional[str] = None
    detected_last_word: str = ""
    extension: str = ""
    error_message: Optional[str] = None


@dataclass
class ImageGroup:
    """Represents a grouped set of up to 4 channel images sharing a filename prefix."""
    group_id: str
    prefix: str
    files_by_channel: Dict[str, ParsedFileItem] = field(default_factory=dict)
    is_split_overflow: bool = False
    overflow_note: Optional[str] = None

    @property
    def available_channels(self) -> List[str]:
        """Return present channels in canonical order."""
        return [ch for ch in ALLOWED_CHANNELS if ch in self.files_by_channel]

    @property
    def file_count(self) -> int:
        return len(self.files_by_channel)


@dataclass
class BatchGroupingResult:
    """Summary of parsing and grouping a batch of uploaded files."""
    groups: List[ImageGroup] = field(default_factory=list)
    invalid_files: List[ParsedFileItem] = field(default_factory=list)
    overflow_errors: List[str] = field(default_factory=list)
    total_uploaded: int = 0


def parse_microscope_filename(filename: str, file_obj: Any = None) -> ParsedFileItem:
    """
    Parse a microscope image filename into (prefix, channel) strictly based on the
    last word before the file extension.

    Allowed color keywords (case-insensitive):
    - 'bright field' (also accepts 'bright_field' / 'bright-field' / 'brightfield')
    - 'blue'
    - 'green'
    - 'red'
    """
    base_name = os.path.basename(filename.strip())
    stem, ext = os.path.splitext(base_name)
    stem_clean = stem.strip()

    if not stem_clean:
        return ParsedFileItem(
            filename=base_name,
            file_obj=file_obj,
            is_valid=False,
            prefix="",
            channel=None,
            detected_last_word="(empty)",
            extension=ext,
            error_message=(
                f"Invalid filename **'{base_name}'**: filename stem is empty. "
                f"Please rename the file to end with one of the allowed color keywords: "
                f"`bright field`, `blue`, `green`, or `red` (e.g., `Sample1_blue{ext or '.tif'}`)."
            ),
        )

    # 1. Check for 'bright field' at the end of the stem
    bf_match = _BRIGHT_FIELD_PATTERN.match(stem_clean)
    if bf_match:
        raw_prefix = bf_match.group(1).rstrip(" _-.")
        normalized_prefix = raw_prefix if raw_prefix else "(No Prefix)"
        return ParsedFileItem(
            filename=base_name,
            file_obj=file_obj,
            is_valid=True,
            prefix=normalized_prefix,
            channel="bright field",
            detected_last_word="bright field",
            extension=ext,
            error_message=None,
        )

    # 2. Extract the last word of the stem
    word_match = _LAST_WORD_PATTERN.match(stem_clean)
    if word_match:
        raw_prefix = word_match.group(1).rstrip(" _-.")
        last_word = word_match.group(2)
    else:
        raw_prefix = ""
        last_word = stem_clean

    last_word_lower = last_word.lower()

    # Check if the user wrote 'field' preceded by something else that wasn't 'bright',
    # or any word other than blue, green, red
    if last_word_lower in ("blue", "green", "red"):
        normalized_prefix = raw_prefix if raw_prefix else "(No Prefix)"
        return ParsedFileItem(
            filename=base_name,
            file_obj=file_obj,
            is_valid=True,
            prefix=normalized_prefix,
            channel=last_word_lower,
            detected_last_word=last_word,
            extension=ext,
            error_message=None,
        )

    # Construct helpful rename example
    example_prefix = raw_prefix if raw_prefix else stem_clean
    suggested_name = f"{example_prefix}_blue{ext or '.tif'}"

    return ParsedFileItem(
        filename=base_name,
        file_obj=file_obj,
        is_valid=False,
        prefix=raw_prefix,
        channel=None,
        detected_last_word=last_word,
        extension=ext,
        error_message=(
            f"Invalid color keyword in **'{base_name}'**: the last word before the extension is "
            f"**'{last_word}'**, which is not one of the four allowed color keywords "
            f"(`bright field`, `blue`, `green`, `red`). "
            f"Please rename the file with the correct color at the end (for example: `{suggested_name}`)."
        ),
    )


def group_uploaded_files(
    uploaded_items: List[Tuple[str, Any]],
    split_overflow: bool = True,
) -> BatchGroupingResult:
    """
    Parse and group a batch of (filename, file_obj) pairs by their exact filename prefix.

    Grouping rules:
    - Files with an unrecognized last word are collected in `invalid_files` so the UI
      can immediately display `st.error` for each one.
    - Valid files with the exact same prefix belong to the same group.
    - A group can have a maximum of 4 files (one for each allowed channel:
      'bright field', 'blue', 'green', 'red').
    - If 1, 2, 3, or 4 files match a prefix (with distinct channels), they are grouped normally.
    - If a prefix has more than 4 files (or duplicate files for the same channel),
      an error message is recorded in `overflow_errors` AND (when `split_overflow=True`)
      the overflow files are split into sequential sub-groups of at most 4 files.
    """
    result = BatchGroupingResult(total_uploaded=len(uploaded_items))
    by_prefix: Dict[str, List[ParsedFileItem]] = {}

    for fname, fobj in uploaded_items:
        parsed = parse_microscope_filename(fname, fobj)
        if not parsed.is_valid:
            result.invalid_files.append(parsed)
        else:
            by_prefix.setdefault(parsed.prefix, []).append(parsed)

    # Sort prefixes alphabetically / naturally for deterministic UI presentation
    for prefix in sorted(by_prefix.keys(), key=lambda s: s.lower()):
        items = by_prefix[prefix]

        # Check if > 4 files share this prefix
        has_more_than_4 = len(items) > 4

        # Check if there are duplicate channels within this prefix
        seen_channels = [item.channel for item in items if item.channel]
        has_duplicate_channel = len(seen_channels) != len(set(seen_channels))

        if has_more_than_4:
            result.overflow_errors.append(
                f"Prefix **'{prefix}'** has **{len(items)} files**, which exceeds the maximum of "
                f"4 files per group (one per allowed channel: `bright field`, `blue`, `green`, `red`). "
                + (
                    "The extra files have been automatically split into separate sub-groups below."
                    if split_overflow
                    else "Please rename or remove duplicate files for this prefix."
                )
            )
        elif has_duplicate_channel:
            result.overflow_errors.append(
                f"Prefix **'{prefix}'** contains multiple files with the same channel keyword "
                f"({len(items)} files total). "
                + (
                    "Duplicate channels have been split into separate sub-groups so each group has at most 1 file per channel."
                    if split_overflow
                    else "Each group may only contain at most one file per channel."
                )
            )

        if not has_more_than_4 and not has_duplicate_channel:
            # Normal group of 1, 2, 3, or 4 files
            ch_map: Dict[str, ParsedFileItem] = {}
            for item in items:
                if item.channel is not None:
                    ch_map[item.channel] = item
            result.groups.append(
                ImageGroup(
                    group_id=prefix,
                    prefix=prefix,
                    files_by_channel=ch_map,
                    is_split_overflow=False,
                )
            )
        else:
            if not split_overflow:
                # Keep only the first valid up-to-4 unique channels or skip
                continue

            # Split items into sub-groups where each sub-group has at most 4 files (1 per channel)
            sub_groups: List[Dict[str, ParsedFileItem]] = []
            for item in items:
                ch = item.channel
                if ch is None:
                    continue
                placed = False
                for sg in sub_groups:
                    if ch not in sg and len(sg) < 4:
                        sg[ch] = item
                        placed = True
                        break
                if not placed:
                    sub_groups.append({ch: item})

            for idx, sg in enumerate(sub_groups, start=1):
                gid = f"{prefix} [Batch {idx}/{len(sub_groups)}]"
                result.groups.append(
                    ImageGroup(
                        group_id=gid,
                        prefix=prefix,
                        files_by_channel=sg,
                        is_split_overflow=True,
                        overflow_note=(
                            f"Split sub-group {idx} of {len(sub_groups)} for prefix '{prefix}' "
                            f"({len(items)} files uploaded with this prefix)."
                        ),
                    )
                )

    return result
