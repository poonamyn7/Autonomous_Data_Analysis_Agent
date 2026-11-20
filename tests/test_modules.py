"""Automated tests. Run from the project folder with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.agent import analyze_dataset
from modules.data_cleaner import clean_data
from modules.pipeline import run_pipeline
from modules.question_answering import answer_question
from modules.utils import get_id_columns, get_numeric_columns


def make_students(n):
    rng = np.random.default_rng(0)
    study = rng.normal(5.5, 2, n).clip(1, 10).round(1)
    return pd.DataFrame({
        "Student_ID": range(1, n + 1),
        "Age": rng.integers(18, 25, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Study_Hours": study,
        "Attendance": rng.normal(85, 9, n).clip(55, 100).round(0),
        "Marks": (30 + 6 * study + rng.normal(0, 4, n)).clip(0, 100).round(0),
    })


def test_id_columns_are_detected_and_excluded():
    df = make_students(50)
    assert get_id_columns(df) == ["Student_ID"]
    assert "Student_ID" not in get_numeric_columns(df)


def test_cleaner_fixes_messy_data():
    raw = pd.DataFrame({
        " Unit Price ": ["10", "20", "N/A", "40", "50", "60"],
        "Region": ["N", "S", " ", "N", "S", "E"],
        "Qty": [1, 2, 3, 4, 5, 6],
    })
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # exact duplicate
    clean = clean_data(raw)
    assert "Unit_Price" in clean.columns
    assert pd.api.types.is_numeric_dtype(clean["Unit_Price"])
    assert clean.isna().sum().sum() == 0
    assert len(clean) == len(raw) - 1
    assert clean.attrs["missing_before"]["Unit_Price"] == 1


def test_agent_skips_ml_on_small_data_and_runs_on_large():
    small = analyze_dataset(clean_data(make_students(15)))
    large = analyze_dataset(clean_data(make_students(100)))
    assert not small["run_ml"] and "15 rows" in small["ml_skip_reason"]
    assert large["run_ml"] and large["target_column"] == "Marks"


def test_target_override():
    plan = analyze_dataset(clean_data(make_students(100)), target_override="Attendance")
    assert plan["target_column"] == "Attendance"


@pytest.mark.parametrize("rows", [15, 100])
def test_full_pipeline(rows):
    result = run_pipeline(make_students(rows))
    assert result["pdf_bytes"].startswith(b"%PDF")
    assert result["execution"]["ml_error"] is None
    assert "**" in result["insights"]
    if rows >= 30:
        assert result["execution"]["ml"]["r2"] > 0.5
    else:
        assert result["execution"]["ml"] is None


def test_pipeline_survives_text_only_data():
    df = pd.DataFrame({"name": list("abcdefghij"), "city": ["x", "y"] * 5})
    result = run_pipeline(df)
    assert result["pdf_bytes"].startswith(b"%PDF")
    assert result["execution"]["ml"] is None
    assert result["validation"]["warnings"]


def test_question_answering():
    result = run_pipeline(make_students(100))
    ar = result["analysis_results"]
    df = ar["dataframe"]

    answer, decision = answer_question("what is the average marks", ar)
    assert f"{df['Marks'].mean():,.2f}" in answer and decision["type"] == "Statistic"

    answer, _ = answer_question("how many rows", ar)
    assert "100 rows" in answer

    answer, decision = answer_question("average marks by gender", ar)
    assert decision["type"] == "Group comparison" and "Female" in answer

    answer, decision = answer_question("correlation between study hours and marks", ar)
    assert decision["type"] == "Correlation"

    _, decision = answer_question("asdf qwerty", ar)
    assert decision["type"] == "Unknown"


# ---------------------------------------------------------------- multi-file loading
import io
import zipfile

from modules.file_loader import (
    combine_datasets,
    group_by_schema,
    load_folder,
    load_uploaded_files,
)


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._data = name, data

    def getvalue(self):
        return self._data


def _csv(df):
    return df.to_csv(index=False).encode()


def test_load_folder_searches_subfolders(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".hidden").mkdir()
    make_students(5).to_csv(tmp_path / "top.csv", index=False)
    make_students(6).to_csv(tmp_path / "a" / "b" / "deep.csv", index=False)
    make_students(7).to_csv(tmp_path / ".hidden" / "skip.csv", index=False)
    (tmp_path / "a" / "empty.csv").write_text("x,y\n")
    datasets, problems = load_folder(str(tmp_path))
    assert sorted(datasets) == ["a/b/deep.csv", "top.csv"]
    assert any("empty.csv" in p for p in problems)
    assert load_folder(str(tmp_path / "nope"))[1]


def test_load_zip_with_folders_and_duplicate_names():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("data/jan/s.csv", _csv(make_students(5)))
        z.writestr("data/feb/s.csv", _csv(make_students(6)))
        z.writestr("__MACOSX/data/._s.csv", b"junk")
        z.writestr("data/readme.txt", b"hi")
    datasets, problems = load_uploaded_files([FakeUpload("data.zip", buffer.getvalue())])
    assert sorted(datasets) == ["data/feb/s.csv", "data/jan/s.csv"]
    assert not problems


def test_loose_files_with_same_name_are_kept_separate():
    files = [FakeUpload("s.csv", _csv(make_students(5))), FakeUpload("s.csv", _csv(make_students(6)))]
    datasets, _ = load_uploaded_files(files)
    assert len(datasets) == 2


def test_bad_inputs_are_reported_not_raised():
    datasets, problems = load_uploaded_files([
        FakeUpload("bad.zip", b"not a zip"),
        FakeUpload("empty.csv", b"a,b\n"),
    ])
    assert not datasets and len(problems) == 2


def test_semicolon_and_latin1_files():
    datasets, _ = load_uploaded_files([
        FakeUpload("semi.csv", b"x;y\n1;2\n3;4\n"),
        FakeUpload("latin.csv", "city,n\nZ\xfcrich,1\n".encode("latin-1")),
    ])
    assert datasets["semi.csv"].shape == (2, 2)
    assert datasets["latin.csv"].iloc[0, 0] == "Zürich"


def test_combine_and_group_by_schema():
    a, b = make_students(10), make_students(12)
    c = make_students(8).assign(Extra=1)
    combined, notes = combine_datasets({"a.csv": a, "b.csv": b})
    assert len(combined) == 22 and not notes
    assert set(combined["Source_File"]) == {"a.csv", "b.csv"}

    _, notes = combine_datasets({"a.csv": a, "c.csv": c})
    assert notes and "Extra" in notes[0]

    _, notes = combine_datasets({"a.csv": a, "z.csv": pd.DataFrame({"q": [1, 2]})})
    assert "no common columns" in notes[0]

    groups = group_by_schema({"a.csv": a, "b.csv": b, "c.csv": c})
    assert [len(g) for g in groups.values()] == [2, 1]


def test_combined_files_run_through_pipeline():
    combined, _ = combine_datasets({"a.csv": make_students(40), "b.csv": make_students(40)})
    result = run_pipeline(combined)
    assert result["profile"]["rows"] == 80
    assert result["pdf_bytes"].startswith(b"%PDF")
    answer, _ = answer_question("average marks by source file", result["analysis_results"])
    assert "a.csv" in answer and "b.csv" in answer
