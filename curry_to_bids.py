#!/usr/bin/env python3
"""Convert CURRY EEG data (.cdt) to BIDS format using MNE-Python and MNE-BIDS.

Usage:
    python curry_to_bids.py --dry-run              # preview conversions
    python curry_to_bids.py --subjects SA076       # convert one subject
    python curry_to_bids.py --overwrite             # convert all, overwrite existing
    python curry_to_bids.py --log-file convert.log  # save log to file
"""

import argparse
import logging
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "DATA"
BIDS_ROOT = Path(__file__).resolve().parent

SUBJECT_IDS = ["SA076", "SA088", "SA095"]

# BIDS task labels must be alphanumeric (no hyphens)
TASK_MAP = {
    "cpt-aud":    "cptaud",
    "cpt-vis":    "cptvis",
    "pvt-aud":    "pvtaud",
    "pvt-vis":    "pvtvis",
    "sst-aud":    "sstaud",
    "sst-vis":    "sstvis",
    "stroop-aud": "stroopaud",
    "stroop-vis": "stroopvis",
    "rest-ec":    "restec",
    "rest-eo":    "resteo",
}

REST_TASKS = {"rest-ec", "rest-eo"}

DATASET_NAME = "EEG Cognitive Tasks Dataset"
BIDS_VERSION = "1.9.0"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("curry_to_bids")


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# File discovery and parsing
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(
    r"^(?P<subject>SA\d{3})_(?P<task>[a-z]+-[a-z]{2,3})\.cdt$"
)


def get_curry_dir(subject_id: str) -> Path:
    return DATA_ROOT / subject_id / f"{subject_id} CURRY"


def parse_filename(cdt_path: Path) -> dict | None:
    """Parse a CURRY .cdt filename into its components."""
    m = _FILENAME_RE.match(cdt_path.name)
    if not m:
        return None
    task_raw = m.group("task")
    if task_raw not in TASK_MAP:
        logger.warning("Unknown task '%s' in %s", task_raw, cdt_path.name)
        return None
    ceo_path = cdt_path.parent / (cdt_path.name + ".ceo")
    return {
        "subject_id": m.group("subject"),
        "task_raw":   task_raw,
        "task_bids":  TASK_MAP[task_raw],
        "is_rest":    task_raw in REST_TASKS,
        "cdt_path":   cdt_path,
        "has_events": ceo_path.exists(),
    }


def discover_files(subject_id: str) -> list[dict]:
    """Find all .cdt files for a subject and return parsed descriptors."""
    curry_dir = get_curry_dir(subject_id)
    if not curry_dir.exists():
        logger.warning("CURRY directory not found: %s", curry_dir)
        return []
    records = []
    for cdt_file in sorted(curry_dir.glob("*.cdt")):
        info = parse_filename(cdt_file)
        if info:
            records.append(info)
        else:
            logger.warning("Could not parse filename: %s", cdt_file.name)
    return records


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(all_records: list[dict]) -> None:
    """Print a preview of what would be converted."""
    print("\n=== DRY RUN: Files that would be converted ===\n")
    for rec in all_records:
        flags = []
        if rec["has_events"]:
            flags.append("events")
        if rec["is_rest"]:
            flags.append("rest")
        bids_name = (
            f"sub-{rec['subject_id']}/eeg/"
            f"sub-{rec['subject_id']}_task-{rec['task_bids']}_eeg.vhdr"
        )
        print(f"  {rec['cdt_path'].name:40s} -> {bids_name}  [{', '.join(flags)}]")
    print(f"\nTotal: {len(all_records)} files\n")


# ---------------------------------------------------------------------------
# BIDS conversion (imports deferred so --dry-run works without MNE)
# ---------------------------------------------------------------------------


def convert_file(rec: dict, overwrite: bool = False) -> bool:
    """Read one CURRY file and write it to BIDS. Returns True on success."""
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    # Read CURRY data
    try:
        raw = mne.io.read_raw_curry(rec["cdt_path"], preload=False, verbose="WARNING")
    except Exception as exc:
        logger.error("Failed to read %s: %s", rec["cdt_path"].name, exc)
        return False

    # Extract events (if not a resting-state task)
    events = None
    event_id = None
    if not rec["is_rest"] and rec["has_events"]:
        try:
            events, event_id = mne.events_from_annotations(raw, verbose="WARNING")
            logger.info("  Events: %d total, %d unique codes", len(events), len(event_id))
        except Exception as exc:
            logger.warning("  Could not extract events: %s", exc)

    # Build BIDS path
    bids_path = BIDSPath(
        subject=rec["subject_id"],
        task=rec["task_bids"],
        datatype="eeg",
        root=BIDS_ROOT,
        suffix="eeg",
        extension=".vhdr",
    )

    # Write BIDS
    try:
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            events=events,
            event_id=event_id,
            overwrite=overwrite,
            allow_preload=True,
            format="BrainVision",
            verbose="WARNING",
        )
        logger.info("  Written -> %s", bids_path.fpath.relative_to(BIDS_ROOT))
        return True
    except Exception as exc:
        logger.error("  Failed to write BIDS for %s: %s", rec["cdt_path"].name, exc)
        return False
    finally:
        raw.close()


def write_dataset_description() -> None:
    """Write dataset_description.json to BIDS root."""
    from mne_bids import make_dataset_description

    make_dataset_description(
        path=BIDS_ROOT,
        name=DATASET_NAME,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CURRY EEG data (.cdt) to BIDS format"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview conversions without writing files",
    )
    p.add_argument(
        "--subjects", nargs="+", default=SUBJECT_IDS,
        help="Subjects to process (default: all)",
    )
    p.add_argument(
        "--tasks", nargs="+", default=None,
        help="Tasks to process, e.g. cpt-aud rest-ec (default: all)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing BIDS files",
    )
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="Path to save log file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    logger.info("CURRY-to-BIDS conversion")
    logger.info("  Source: %s", DATA_ROOT)
    logger.info("  Output: %s", BIDS_ROOT)

    # Discover files
    all_records: list[dict] = []
    for subj in args.subjects:
        records = discover_files(subj)
        if args.tasks:
            records = [r for r in records if r["task_raw"] in args.tasks]
        all_records.extend(records)
        logger.info("%s: %d CDT files found", subj, len(records))

    if not all_records:
        logger.warning("No files to process.")
        return

    # Dry run mode
    if args.dry_run:
        dry_run(all_records)
        return

    # Convert
    write_dataset_description()

    success, failure = 0, 0
    for rec in all_records:
        logger.info("Processing: %s", rec["cdt_path"].name)
        if convert_file(rec, overwrite=args.overwrite):
            success += 1
        else:
            failure += 1

    logger.info("Done. Success: %d, Failed: %d", success, failure)


if __name__ == "__main__":
    main()
