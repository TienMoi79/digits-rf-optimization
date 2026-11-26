"""Reject uncommitted source while accepting Git-normalized Windows line endings."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.package_submission import require_clean_tree


class SubmissionTests(unittest.TestCase):
    def fixture(self, root):
        def git(*args):
            subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        git("init", "-q")
        git("config", "user.name", "Packaging Test")
        git("config", "user.email", "packaging-test@example.invalid")
        git("config", "core.autocrlf", "true")
        (root / ".gitattributes").write_bytes(b"*.txt text eol=lf\n")
        (root / "source.txt").write_bytes(b"original\ncontent\n")
        git("add", ".")
        git("commit", "-qm", "Temporary packaging test fixture")
        return git

    def test_line_ending_rewrite_does_not_count_as_a_source_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            (root / "source.txt").write_bytes(b"original\r\ncontent\r\n")
            require_clean_tree(root)

    def test_modified_staged_and_untracked_files_are_rejected(self):
        for mode in ["modified", "staged", "untracked"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                git = self.fixture(root)
                if mode == "untracked":
                    (root / "extra.txt").write_bytes(b"not committed\n")
                else:
                    (root / "source.txt").write_bytes(b"different content\n")
                    if mode == "staged":
                        git("add", "source.txt")
                with self.assertRaisesRegex(ValueError, "Commit"):
                    require_clean_tree(root)
