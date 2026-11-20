"""Runs app.py against a fake Streamlit so the app logic can be checked without a browser.

Usage:  python tests/smoke_app.py [--q QUESTION] [--files a.csv b.zip ...] [--folder PATH] [--mode 0|1]
(--mode 0 = combine all, 1 = same-columns group, 2 = one file; default = the app's own default)
"""
import argparse
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--q", default="")
ap.add_argument("--files", nargs="*", default=[])
ap.add_argument("--folder", default="")
ap.add_argument("--mode", type=int, default=-1)
args = ap.parse_args()
question = args.q


class Stop(Exception):
    pass


calls = {"error": [], "warning": [], "sidebar_warning": [], "download": [], "markdown": [], "pyplot": 0}
st = MagicMock()
st.cache_resource = lambda **kw: (lambda f: f)
st.cache_data = lambda **kw: (lambda f: f)
st.columns = lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs = lambda labels: [MagicMock() for _ in labels]
st.selectbox = lambda label, options, **kw: options[0]

st.text_input = lambda label, *a, **kw: args.folder if "folder" in label.lower() else question
st.radio = lambda label, options, index=0, **kw: options[args.mode if args.mode >= 0 else index]
st.stop = MagicMock(side_effect=Stop)
st.error = lambda m, *a, **k: calls["error"].append(m)
st.warning = lambda m, *a, **k: calls["warning"].append(m)
st.markdown = lambda m, *a, **k: calls["markdown"].append(m)
st.download_button = lambda **kw: calls["download"].append((kw["file_name"], len(kw["data"])))
st.pyplot = lambda fig, *a, **k: calls.__setitem__("pyplot", calls["pyplot"] + 1)

import io


class FakeUpload(io.BytesIO):
    def __init__(self, path):
        super().__init__(Path(path).read_bytes())
        self.name = Path(path).name


uploads = [FakeUpload(f) for f in args.files]
st.file_uploader = lambda *a, **k: uploads

st.sidebar.warning = lambda m, *a, **k: calls["sidebar_warning"].append(m)
sys.modules["streamlit"] = st
try:
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    print("APP RAN OK")
except Stop:
    print("APP STOPPED (st.stop) errors:", calls["error"])
print("sidebar warnings:", calls["sidebar_warning"])
print("page warnings:", [w[:110] for w in calls["warning"]][:3])
print("pyplot calls:", calls["pyplot"], "| downloads:", calls["download"], "| warnings:", len(calls["warning"]))
if question and calls["markdown"]:
    print("answer shown:", calls["markdown"][-1][:300])
