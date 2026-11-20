"""Load one or many CSV files - single files, many files, ZIP archives or whole folders.

Every loader returns ``(datasets, problems)``:
* ``datasets`` - dict mapping a readable file name (with its sub-folder path) to a DataFrame
* ``problems`` - list of plain-English messages about files that could not be used
"""

import io
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

SOURCE_COLUMN = "Source_File"
MAX_FILES = 300
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def read_csv_bytes(data):
    """Parse CSV bytes, trying common encodings and (if needed) other separators."""
    last_error = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
            if frame.shape[1] == 1:  # maybe ';' or tab separated
                try:
                    sniffed = pd.read_csv(
                        io.BytesIO(data), encoding=encoding, sep=None, engine="python"
                    )
                    if sniffed.shape[1] > 1:
                        frame = sniffed
                except Exception:
                    pass
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def _unique(name, datasets):
    if name not in datasets:
        return name
    stem, suffix, i = Path(name).stem, Path(name).suffix, 2
    parent = str(Path(name).parent)
    prefix = "" if parent == "." else parent + "/"
    while f"{prefix}{stem} ({i}){suffix}" in datasets:
        i += 1
    return f"{prefix}{stem} ({i}){suffix}"


def _add(datasets, problems, name, data):
    try:
        frame = read_csv_bytes(data)
    except Exception as error:
        problems.append(f"{name}: could not be read as a CSV ({error}).")
        return
    if frame.empty or frame.shape[1] == 0:
        problems.append(f"{name}: has no data rows, skipped.")
        return
    datasets[_unique(name, datasets)] = frame


def _is_junk(path):
    parts = PurePosixPath(path).parts
    return any(p == "__MACOSX" or p.startswith(".") for p in parts) or parts[-1].startswith("~$")


def load_uploaded_files(files):
    """Handle Streamlit uploads: any mix of .csv and .zip (zips may contain sub-folders)."""
    datasets, problems = {}, []
    for upload in files or []:
        name = upload.name
        data = upload.getvalue()
        if name.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                problems.append(f"{name}: is not a valid ZIP file.")
                continue
            members = [
                m for m in archive.namelist()
                if m.lower().endswith(".csv") and not m.endswith("/") and not _is_junk(m)
            ]
            if not members:
                problems.append(f"{name}: contains no CSV files.")
            for member in members[:MAX_FILES]:
                _add(datasets, problems, member, archive.read(member))
            if len(members) > MAX_FILES:
                problems.append(f"{name}: only the first {MAX_FILES} CSV files were loaded.")
        else:
            _add(datasets, problems, name, data)
    return datasets, problems


def load_folder(folder):
    """Load every CSV inside ``folder`` and all of its sub-folders (local runs only)."""
    datasets, problems = {}, []
    root = Path(str(folder).strip().strip('"').strip("'")).expanduser()
    if not root.is_dir():
        return datasets, [f"Folder not found: {root}"]
    paths = sorted(
        p for p in root.rglob("*.csv")
        if p.is_file() and not _is_junk(p.relative_to(root).as_posix())
    )
    if not paths:
        problems.append(f"No CSV files were found in {root} or its sub-folders.")
    for path in paths[:MAX_FILES]:
        try:
            data = path.read_bytes()
        except OSError as error:
            problems.append(f"{path.name}: could not be opened ({error}).")
            continue
        _add(datasets, problems, path.relative_to(root).as_posix(), data)
    if len(paths) > MAX_FILES:
        problems.append(f"Only the first {MAX_FILES} of {len(paths)} CSV files were loaded.")
    return datasets, problems


def combine_datasets(datasets):
    """Stack all files into one DataFrame and add a ``Source_File`` column.

    Returns ``(dataframe, notes)``; notes explain anything the person should know,
    for example that files had different columns.
    """
    notes = []
    frames = []
    for name, frame in datasets.items():
        frame = frame.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        frame[SOURCE_COLUMN] = name
        frames.append(frame)

    column_sets = [set(f.columns) - {SOURCE_COLUMN} for f in frames]
    common = set.intersection(*column_sets)
    union = set.union(*column_sets)
    if not common:
        notes.append(
            "These files share no common columns, so combining them is not meaningful. "
            "Choose 'Analyse one file' instead."
        )
    elif union != common:
        extra = sorted(union - common)
        notes.append(
            f"The files do not all have the same columns. {len(common)} column(s) are shared; "
            f"these appear in only some files: {', '.join(extra[:8])}"
            f"{'...' if len(extra) > 8 else ''}. Gaps are filled automatically during cleaning."
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, notes


def group_by_schema(datasets):
    """Group files that have exactly the same columns.

    Returns a dict ``label -> {file name: DataFrame}``, biggest group (most rows) first.
    """
    groups = {}
    for name, frame in datasets.items():
        key = tuple(sorted(str(c).strip() for c in frame.columns))
        groups.setdefault(key, {})[name] = frame
    ordered = sorted(groups.values(), key=lambda g: -sum(len(f) for f in g.values()))
    labelled = {}
    for members in ordered:
        first = next(iter(members.values()))
        cols = ", ".join(map(str, first.columns))
        cols = cols if len(cols) <= 60 else cols[:57] + "..."
        rows = sum(len(f) for f in members.values())
        labelled[f"{len(members)} file(s), {rows} rows - columns: {cols}"] = members
    return labelled


def summarize(datasets):
    """A small table describing every loaded file."""
    return pd.DataFrame(
        [
            {"File": name, "Rows": len(f), "Columns": f.shape[1], "Column names": ", ".join(map(str, f.columns))}
            for name, f in datasets.items()
        ]
    )
