"""Move the industrial lessons into the single, sequential course notebook tree."""
from __future__ import annotations

import hashlib
import argparse
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "courses" / "industrial-vision-lab" / "notebooks"
DESTINATION = ROOT / "notebooks"
DRAFT = DESTINATION / "_drafts" / "Untitled.ipynb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-generated", action="store_true")
    args = parser.parse_args()
    files = sorted(SOURCE.glob("*.ipynb"))
    if len(files) != 11:
        raise RuntimeError(f"Expected 11 industrial source notebooks, found {len(files)}")
    mappings = []
    for index, source in enumerate(files, start=22):
        notebook = nbformat.read(source, as_version=4)
        prefix = f"{index:02d}"
        first_code = True
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                def shift_heading(match):
                    old = int(match.group(2))
                    return f"{match.group(1)}{old + 22:02d}{match.group(3)}"
                cell.source = re.sub(r"(?m)^(#{1,6}\s+)(\d{2})([｜|])", shift_heading, cell.source)
            elif cell.cell_type == "code":
                if first_code:
                    cell.source = cell.source.replace(
                        'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "lab_utils.py").is_file())',
                        'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_utils.py").is_file())\n'
                        'INDUSTRIAL_ROOT = ROOT / "courses" / "industrial-vision-lab"\n'
                        'if str(INDUSTRIAL_ROOT) not in sys.path:\n    sys.path.insert(0, str(INDUSTRIAL_ROOT))',
                    )
                    first_code = False
                cell.source = cell.source.replace(
                    'ROOT / "runtime/', 'INDUSTRIAL_ROOT / "runtime/'
                ).replace('ROOT / "outputs/', 'INDUSTRIAL_ROOT / "outputs/')
        notebook.metadata.setdefault("industrial_vision_lab", {})["unified_course_number"] = index
        target = DESTINATION / f"{prefix}_{source.name[3:]}"
        if target.exists() and not args.overwrite_generated:
            raise FileExistsError(f"Refusing to overwrite {target}")
        nbformat.write(notebook, target)
        mappings.append((source, target))

    if (ROOT / "Untitled.ipynb").exists():
        if DRAFT.exists():
            raise FileExistsError(f"Refusing to overwrite {DRAFT}")
        DRAFT.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / "Untitled.ipynb").replace(DRAFT)

    for source, target in mappings:
        print(f"{source.name} -> {target.relative_to(ROOT)} source={digest(source)} migrated={digest(target)}")


if __name__ == "__main__":
    main()
