import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PRISTI_DIR = REPO_ROOT / "PriSTI"

# Known-good commit that contains required code
PRISTI_REMOTE = "https://github.com/LMZZML/PriSTI.git"
PRISTI_COMMIT = "8ba37cc1a84cf706e767ae071101b3f02c921c9a"

def ensure_pristi_present():
    """
    Make sure PriSTI exists and has main_model.py and diff_models.py.
    Try submodule update first; if it fails or files are missing, clone and checkout a known commit.
    """
    # 1) Try submodule update if .gitmodules exists and path looks clean
    gm = REPO_ROOT / ".gitmodules"
    tried_submodule = False
    if gm.exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            tried_submodule = True
        except subprocess.CalledProcessError as e:
            print("WARN: submodule update failed; will fall back to direct clone.\n", e.stdout)

    # 2) If still missing or broken, do a direct clone/checkout
    need_clone = (
        not PRISTI_DIR.exists()
        or not (PRISTI_DIR / "main_model.py").exists()
        or not (PRISTI_DIR / "diff_models.py").exists()
    )

    if need_clone:
        # Remove broken dir if present
        if PRISTI_DIR.exists():
            import shutil
            shutil.rmtree(PRISTI_DIR, ignore_errors=True)

        print("INFO: Cloning PriSTI directly (fallback)...")
        subprocess.run(["git", "clone", PRISTI_REMOTE, str(PRISTI_DIR)], check=True)

        # Checkout pinned commit (or fallback to main if needed)
        try:
            subprocess.run(["git", "fetch", "origin"], cwd=PRISTI_DIR, check=True)
            subprocess.run(["git", "checkout", PRISTI_COMMIT], cwd=PRISTI_DIR, check=True)
        except subprocess.CalledProcessError:
            print(f"WARN: Commit {PRISTI_COMMIT} not available; falling back to 'main'.")
            subprocess.run(["git", "checkout", "main"], cwd=PRISTI_DIR, check=True)

    # Final sanity check
    missing = [p for p in ["main_model.py", "diff_models.py"] if not (PRISTI_DIR / p).exists()]
    if missing:
        raise RuntimeError(f"PriSTI is missing required files: {missing}. Please verify the repository state.")

def set_pythonpath_for_streamlit():
    """
    Ensure the Streamlit subprocess inherits a PYTHONPATH that includes:
    - PriSTI directory (for diff_models, etc.)
    - Repo root (for local imports)
    """
    paths = [str(PRISTI_DIR), str(REPO_ROOT)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)

def main():
    ensure_pristi_present()
    set_pythonpath_for_streamlit()

    # Debug print to verify env passed to subprocess
    print("=== DEBUG: PYTHONPATH for Streamlit ===")
    for i, p in enumerate(os.environ["PYTHONPATH"].split(os.pathsep), 1):
        print(f"{i}: {p}")
    print("=======================================")

    # Launch Streamlit; subprocess inherits our PYTHONPATH
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.runOnSave=false", "--logger.level=error"
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

if __name__ == "__main__":
    main()
