"""Package the clean committed project without environments, caches or Git metadata."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "digits-rf-optimization/"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "node_modules"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".ipynb", ".txt", ".yml", ".yaml", ".ps1", ".bat", ".js", ".mjs", ".css", ".html"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def canonical(data, path):
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes", "LICENSE"} else data


def require_clean_tree(root):
    # Windows can report a modified stat entry after a validator rewrites LF as
    # CRLF, even when Git's content diff is empty. Check actual source changes;
    # every packed file is still compared to the working tree below.
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root)
    if diff.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect committed source changes.")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    if diff.returncode or untracked:
        raise ValueError("Commit the project changes before packaging.")


def package(output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Place the submission ZIP outside the project folder.")
    checksum = output.with_name(output.name + ".sha256.txt")
    if output.exists() or output.is_symlink() or checksum.exists():
        raise FileExistsError("Submission archive/checksum already exists; use a new output name.")
    require_clean_tree(ROOT)
    commit = git("rev-parse", "HEAD").decode().strip()
    tracked = {name for name in git("ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0") if name}
    for name in tracked:
        p = Path(name)
        if set(p.parts) & EXCLUDED_PARTS or p.suffix in {".pyc", ".tmp", ".zip", ".pem", ".key"} or p.name.startswith((".env", "~$")):
            raise ValueError(f"Unexpected temporary/private file in committed source: {name}")
    required = {"README.md", "docs/nop-bai.md", "src/experiment.py", "configs/experiment.json",
                "results/models/selected.joblib", "results/trials.json", "web/dist/index.html",
                "web/start_web.bat", "web/serve.py", "notebooks/optimization_walkthrough.ipynb",
                "notebooks/robustness_study.ipynb"}
    if not required <= tracked:
        raise ValueError("Missing files required for the submission/demo.")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=zip", f"--prefix={PREFIX}", f"--output={output}", commit], cwd=ROOT, check=True)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        entries = {name[len(PREFIX):] for name in archive.namelist() if not name.endswith("/")}
        if entries != tracked:
            raise ValueError("ZIP content does not match the committed file list.")
        for name in tracked:
            current = canonical((ROOT / name).read_bytes(), Path(name))
            packed = canonical(archive.read(PREFIX + name), Path(name))
            if current != packed:
                raise ValueError(f"ZIP differs from the clean working tree: {name}")
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum.write_text(f"Commit: {commit}\nSHA256: {sha}\nArchive: {output.name}\nFiles: {len(tracked)}\n", encoding="utf-8")
    print(f"PASS: {len(tracked)} committed files; every ZIP file checked; no environment/cache/Git metadata.")
    print(f"Commit: {commit}\nZIP: {output}\nSHA256: {sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
