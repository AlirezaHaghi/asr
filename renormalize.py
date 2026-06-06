"""
renormalize.py
--------------
Re-applies normalize_for_wer to all prediction JSON files and
drops the now-redundant `reference_raw` field.

Usage
-----
  python renormalize.py                        # default: ./atc_asr_output
  python renormalize.py --dir /path/to/output  # custom output dir
  python renormalize.py --dry-run              # preview changes without writing
"""

import argparse
import json
from pathlib import Path

from jiwer import wer, cer
from text_normalizer import normalize_for_wer


def renormalize_file(path: Path, dry_run: bool = False) -> dict:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    n_changed = 0

    for r in records:
        ref_new = normalize_for_wer(r.get("reference_raw") or r["reference"])
        hyp_new = normalize_for_wer(r.get("hypothesis_raw") or r["hypothesis"])

        changed = (ref_new != r["reference"]) or (hyp_new != r["hypothesis"])
        if changed:
            n_changed += 1

        r["reference"] = ref_new
        r["hypothesis"] = hyp_new
        r["wer"] = round(wer(ref_new, hyp_new) if ref_new else 0.0, 4)
        r["cer"] = round(cer(ref_new, hyp_new) if ref_new else 0.0, 4)

        # drop reference_raw — redundant, recoverable from dataset
        r.pop("reference_raw", None)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return {"samples": len(records), "changed": n_changed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     default="atc_asr_output",
                        help="directory containing predictions_run*.json files")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would change without writing")
    args = parser.parse_args()

    output_dir = Path(args.dir)
    pred_files = sorted(output_dir.glob("predictions_run*.json"))

    if not pred_files:
        print(f"No prediction files found in {output_dir}")
        return

    print(f"{'dry-run mode' if args.dry_run else 'updating files in'}: {output_dir}\n")
    print(f"{'File':<40} {'Samples':>8} {'Changed':>8}")
    print("-" * 58)

    for path in pred_files:
        result = renormalize_file(path, dry_run=args.dry_run)
        status = "(not written)" if args.dry_run else "✓"
        print(f"{path.name:<40} {result['samples']:>8} {result['changed']:>8}  {status}")

    print("\nDone.")
    if args.dry_run:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()