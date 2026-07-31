from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "python" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from real_full_matrix import load_corpus


def test_canonical_source_labels_ignore_root_path(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "entry.rs"
    second = second_root / "entry.rs"
    first.write_text("fn stable() {}\n", encoding="utf-8")
    second.write_bytes(first.read_bytes())

    first_corpus, first_digest = load_corpus(
        [first],
        source_labels=["entry.rs"],
    )
    second_corpus, second_digest = load_corpus(
        [second],
        source_labels=["entry.rs"],
    )

    assert first_corpus == second_corpus
    assert first_digest == second_digest
    assert str(first_root) not in first_corpus
    assert str(second_root) not in second_corpus


def test_default_source_identity_remains_path_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first.rs"
    second = tmp_path / "second.rs"
    first.write_text("same\n", encoding="utf-8")
    second.write_bytes(first.read_bytes())

    first_corpus, first_digest = load_corpus([first])
    second_corpus, second_digest = load_corpus([second])

    assert first_corpus != second_corpus
    assert first_digest != second_digest


@pytest.mark.parametrize(
    "labels",
    ([], ["same", "same"], ["", "second"], ["bad\0label", "second"]),
)
def test_invalid_canonical_source_labels_fail(
    tmp_path: Path,
    labels: list[str],
) -> None:
    first = tmp_path / "first.rs"
    second = tmp_path / "second.rs"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_corpus([first, second], source_labels=labels)
