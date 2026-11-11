"""Hard rule 1: the evaluator is frozen. If this fails, someone edited src/trivec/eval/."""

from pathlib import Path

from trivec.eval import frozen


def test_evaluator_is_frozen() -> None:
    problems = frozen.check()
    assert not problems, (
        "src/trivec/eval/ changed: "
        + "; ".join(problems)
        + ". Only an explicit 'unfreeze eval' may update FROZEN.sha256."
    )


def test_check_detects_changes(tmp_path: Path) -> None:
    d = tmp_path / "eval"
    d.mkdir()
    (d / "a.py").write_text("x = 1\n")
    hf = tmp_path / "FROZEN.sha256"
    frozen.write_hashes(d, hf)
    assert frozen.check(d, hf) == []
    (d / "a.py").write_text("x = 2\n")
    assert frozen.check(d, hf) == ["modified: a.py"]
    (d / "b.py").write_text("")
    assert "added: b.py" in frozen.check(d, hf)
    (d / "a.py").unlink()
    assert "deleted: a.py" in frozen.check(d, hf)


def test_line_endings_do_not_matter(tmp_path: Path) -> None:
    d = tmp_path / "eval"
    d.mkdir()
    (d / "a.py").write_bytes(b"x = 1\n")
    hf = tmp_path / "FROZEN.sha256"
    frozen.write_hashes(d, hf)
    (d / "a.py").write_bytes(b"x = 1\r\n")
    assert frozen.check(d, hf) == []
