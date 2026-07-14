"""Static validation for the retrospective ATC-ASR playground archive.

This checks source structure and provenance. It does not run model inference.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


NOTICE = (
    "Retrospective reconstruction created from the final notebook/report. "
    "It is not claimed to be an original timestamped development artifact; "
    "generated metrics require rerunning."
)

EXPECTED = (
    "01_open_atc_datasets_playground",
    "02_dataset_split_leakage_lab",
    "03_audio_io_resampling",
    "04_signal_conditioning",
    "05_atc_text_normalization",
    "06_hotword_callsign_playground",
    "07_silero_vad_playground",
    "08_whisper_playground",
    "09_directory_batch_transcription",
    "10_voxtral_playground",
    "11_wav2vec2_xlsr_playground",
    "12_cross_family_benchmark",
    "13_fastconformer_playground",
    "14_metrics_sdi",
    "15_alignment_error_taxonomy",
    "16_normalization_ablation",
    "17_latency_rtf_profiling",
    "18_parakeet_canary_playground",
)


@dataclass
class Finding:
    level: str
    path: str
    message: str


def has_persian(text: str) -> bool:
    return any(0x600 <= ord(char) <= 0x6FF for char in text)


def parse_python(path: Path, source: str, findings: list[Finding]) -> int:
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        findings.append(Finding("error", str(path), f"Python syntax: {exc}"))
    return len(source.splitlines())


def notebook_report(path: Path, findings: list[Finding]) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", str(path), f"Notebook JSON: {exc}"))
        return {"cells": 0, "code_cells": 0, "code_lines": 0}

    cells = payload.get("cells")
    if payload.get("nbformat") != 4 or not isinstance(cells, list):
        findings.append(Finding("error", str(path), "Expected nbformat 4 with cells"))
        return {"cells": 0, "code_cells": 0, "code_lines": 0}
    if len(cells) < 8:
        findings.append(Finding("error", str(path), f"Only {len(cells)} cells; minimum is 8"))

    combined = []
    code_cells = 0
    code_lines = 0
    markdown_cells = 0
    for index, cell in enumerate(cells):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = str(source)
        combined.append(source)
        if not cell.get("id"):
            findings.append(Finding("error", str(path), f"Cell {index} has no id"))
        if cell.get("cell_type") == "markdown" and source.strip():
            markdown_cells += 1
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        if cell.get("outputs"):
            findings.append(Finding("error", str(path), f"Cell {index} stores output"))
        if cell.get("execution_count") is not None:
            findings.append(Finding("error", str(path), f"Cell {index} stores execution count"))
        clean_lines = [line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))]
        clean = "\n".join(clean_lines)
        code_lines += len(clean_lines)
        if clean.strip():
            parse_python(path, clean, findings)

    text = "\n".join(combined)
    lower = text.lower()
    if "retrospective reconstruction" not in lower or "rerun" not in lower:
        findings.append(Finding("error", str(path), "Missing in-notebook provenance notice"))
    if not has_persian(text):
        findings.append(Finding("warning", str(path), "No Persian explanation/comment"))
    if not code_cells or not markdown_cells:
        findings.append(Finding("error", str(path), "Needs code and markdown cells"))
    if "artifacts/" in text or "artifacts\\" in text:
        findings.append(Finding("error", str(path), "Old artifacts directory convention remains"))
    return {"cells": len(cells), "code_cells": code_cells, "code_lines": code_lines}


def audit(root: Path) -> dict:
    findings: list[Finding] = []
    totals = {
        "experiments": 0,
        "python_files": 0,
        "python_lines": 0,
        "notebooks": 0,
        "notebook_cells": 0,
        "notebook_code_cells": 0,
        "notebook_code_lines": 0,
    }
    layouts = set()

    for name in EXPECTED:
        directory = root / name
        if not directory.is_dir():
            findings.append(Finding("error", str(directory), "Missing experiment directory"))
            continue
        totals["experiments"] += 1
        if (directory / "artifacts").exists():
            findings.append(Finding("error", str(directory / "artifacts"), "Artifact directory must be removed"))

        readme = directory / "README.md"
        requirements = directory / "requirements.txt"
        if not readme.is_file():
            findings.append(Finding("error", str(readme), "Missing README"))
        else:
            readme_text = readme.read_text(encoding="utf-8")
            if NOTICE not in readme_text:
                findings.append(Finding("error", str(readme), "Missing exact reconstruction notice"))
            if not has_persian(readme_text):
                findings.append(Finding("error", str(readme), "README needs Persian explanation"))
            if "artifacts/" in readme_text or "artifacts\\" in readme_text:
                findings.append(Finding("error", str(readme), "Old artifacts path remains"))
        if not requirements.is_file() or not requirements.read_text(encoding="utf-8").strip():
            findings.append(Finding("error", str(requirements), "Missing or empty requirements"))

        scripts = sorted(directory.glob("*.py"))
        notebooks = sorted(directory.glob("*.ipynb"))
        extra_files = [path for path in directory.iterdir() if path.is_file() and path.suffix not in {".py", ".ipynb", ".md", ".txt"}]
        layouts.add((len(scripts), len(notebooks), len(extra_files)))
        if len(scripts) < 2:
            findings.append(Finding("error", str(directory), "Needs at least two Python files"))
        if len(notebooks) < 2:
            findings.append(Finding("error", str(directory), "Needs at least two notebooks"))

        persian_in_scripts = False
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            persian_in_scripts |= has_persian(source)
            lower = source.lower()
            if "retrospective reconstruction" not in lower or "rerun" not in lower:
                findings.append(Finding("error", str(script), "Missing in-script provenance notice"))
            if "artifacts/" in source or "artifacts\\" in source:
                findings.append(Finding("error", str(script), "Old artifacts directory convention remains"))
            totals["python_files"] += 1
            totals["python_lines"] += parse_python(script, source, findings)
        if scripts and not persian_in_scripts:
            findings.append(Finding("warning", str(directory), "No Persian comment in Python files"))

        for notebook in notebooks:
            report = notebook_report(notebook, findings)
            totals["notebooks"] += 1
            totals["notebook_cells"] += report["cells"]
            totals["notebook_code_cells"] += report["code_cells"]
            totals["notebook_code_lines"] += report["code_lines"]

    if len(layouts) < 4:
        findings.append(Finding("warning", str(root), "Bundle layouts are still too uniform"))
    return {
        "archive": str(root.resolve()),
        "validation_scope": "static structure/syntax/provenance; no GPU inference",
        "layout_variants": len(layouts),
        "totals": totals,
        "findings": [asdict(item) for item in findings],
        "ok": not any(item.level == "error" for item in findings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parent)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
