"""Validate and package the reviewed manuscript; no experimental computation."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs/paper/manuscript_v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = BUNDLE / "main.tex"
    text = source.read_text()
    assert source.read_bytes() == (ROOT / "main.tex").read_bytes()
    images = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    assert len(images) == len(set(images)) == 20
    assert all((BUNDLE / "figures" / name).is_file() for name in images)
    assert all(
        digest(path) == digest(ROOT / "figures" / path.name)
        for path in (BUNDLE / "figures").iterdir()
    )
    # The two successful passes are sufficient: pass 2 has stable references.
    # A later redundant third pass hit the terminal runtime limit; its partial
    # output must never replace the preserved complete second-pass PDF.
    log = (BUNDLE / "build/compile_pass2.stdout").read_text(errors="replace")
    assert "Output written on" in log
    problems = [line for line in log.splitlines() if (
        "undefined" in line or "multiply defined" in line
        or line.startswith("!") or "Overfull" in line
        or "Rerun to get cross-references right" in line
    )]
    assert not problems, problems
    shutil.copyfile(BUNDLE / "build/compile_pass2.stdout", BUNDLE / "compile_verification.log")
    shutil.copyfile(BUNDLE / "build/preview_pass2.pdf", BUNDLE / "main.pdf")
    pdf = PdfReader(BUNDLE / "main.pdf")
    provenance_path = BUNDLE / "assembly_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for name, expected in provenance["original_figures"].items():
        assert digest(ROOT / "result/paper_figures/review_v1_913096" / name) == expected
    assert digest(ROOT / provenance["original_backup"]) == provenance["original_main_sha256"]
    provenance["status"] = "compiled_and_packaged"
    provenance["current_main_sha256"] = digest(source)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    check = dict(
        status="passed", pages=len(pdf.pages), figures=len(images),
        tables=text.count(r"\begin{table}"),
        compile_errors=[], unresolved_references=[], overfull_boxes=[],
        original_manuscript_preserved=True, original_figure_job_preserved=True,
        main_sha256=digest(source), pdf_sha256=digest(BUNDLE / "main.pdf"),
        compiler="pdfLaTeX / texlive/20240312", training_runs=0,
        successful_compile_passes=2,
        compile_evidence="compile_verification.log",
        unused_compile_attempt="Redundant third pass interrupted by terminal runtime limit; its partial output is excluded.",
        visual_review_pages=[1, 7, 9, 11, 17],
        remaining_author_actions=["author list and correspondence", "contributions and funding", "repository/access statement", "journal template"],
    )
    (BUNDLE / "validation.json").write_text(json.dumps(check, indent=2) + "\n")
    manifest = dict(
        status="inserted_and_compiled",
        main_sha256=digest(source),
        referenced_images=images,
        assets={p.name: digest(p) for p in sorted((BUNDLE / "figures").iterdir())},
        note="figure_manifest.json is the original job snapshot; this file describes current manuscript copies.",
    )
    (BUNDLE / "manuscript_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [p for p in BUNDLE.iterdir() if p.is_file() and p.suffix != ".zip"]
    files += sorted((BUNDLE / "figures").glob("*"))
    files += sorted((BUNDLE / "tables").glob("*"))
    with zipfile.ZipFile(BUNDLE / "upload_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(BUNDLE))
        assert archive.testzip() is None
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
