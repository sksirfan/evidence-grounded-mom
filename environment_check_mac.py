"""Lightweight environment check for the Intel Mac dashboard-only mode."""
import importlib
import sys
from pathlib import Path

print("\nVERIFIABLE MEETING MINUTES — MAC DASHBOARD CHECK")
print("Python:", sys.version.split()[0])
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    raise SystemExit("Use Python 3.10, 3.11, or 3.12 for this dashboard environment.")

modules = ("streamlit", "networkx", "reportlab")
failed = []
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f"OK  {name}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"FAIL {name}: {exc}")

repo_root = Path(__file__).resolve().parent
print("Package folder:", repo_root)
if failed:
    raise SystemExit("Install requirements_mac.txt and rerun this check.")
print("\nDashboard dependencies are ready. AI processing remains on Google Colab.")
