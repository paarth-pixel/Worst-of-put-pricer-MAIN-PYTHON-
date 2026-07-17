"""End-to-end test: run with  `python tests/test_app.py`  from the repo root.
Uses the fake yfinance in this folder (realistic TSLA/MSFT surface with a
deliberately stale Yahoo IV column) and drives the real app via AppTest."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # app.py / pricing.py
sys.path.insert(0, os.path.dirname(__file__))                      # fake yfinance first
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"),
                       default_timeout=120)
at.run()
at.sidebar.selectbox[1].set_value("MSFT").run()
assert not at.exception, [str(e.value) for e in at.exception]
metrics = {m.label: m.value for m in at.metric}
prem = float([v for k, v in metrics.items() if "analytic" in k.lower()][0].split("%")[0])
assert 6.0 < prem < 9.0, f"premium {prem}% outside expected band for the test surface"
assert any("Cross-check passed" in s.value for s in at.success)
print(f"PASS — premium {prem}%, analytic/MC cross-check green, no exceptions")
