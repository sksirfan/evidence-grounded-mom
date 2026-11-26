# Intel Mac setup — Evidence-Grounded MoM

## Supported split

- **Mac (16 GB RAM):** run the lightweight Streamlit dashboard to inspect meetings already processed in Colab, evidence, verified dependency DAGs, and PDF exports.
- **Google Colab GPU:** run audio transcription, diarization, Qwen generation, evidence retrieval/verification, and Phase 2 dependency extraction.

The current AI stack requires modern PyTorch builds. Official macOS Intel (`x86_64`) wheels are not provided for the project's PyTorch 2.8 stack, so this setup deliberately does not attempt local AI processing.

## 1. Clone the repository

```bash
cd ~/Documents
git clone https://github.com/sksirfan/evidence-grounded-mom.git
cd evidence-grounded-mom
```

If you already cloned it, use that existing folder and pull the latest GitHub changes. Keep a backup of your current `07_dashboard/app.py` before replacing it if you are updating an existing clone.

## 2. Create a Mac dashboard environment

Use Python 3.10, 3.11, or 3.12. From the repository root:

```bash
python3.11 -m venv .venv-mac
source .venv-mac/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_mac.txt
python environment_check_mac.py
```

The files `requirements_mac.txt`, `environment_check_mac.py`, and `config.py` are included at the repository root.

## 3. Connect Google Drive for Desktop

Install/sign in to Google Drive for Desktop and ensure `MTechIndProj/MoM_Project` is available offline or synced. The shared `config.py` tries the standard macOS `~/Library/CloudStorage/GoogleDrive-*/My Drive/MTechIndProj/MoM_Project` location automatically. The dashboard and pipeline use this same configured root.

If your folder is somewhere else, set its exact path before launching Streamlit:

```bash
export MOM_PROJECT_ROOT="$HOME/Library/CloudStorage/GoogleDrive-your-account/My Drive/MTechIndProj/MoM_Project"
```

Replace the example path with the actual path shown in Finder. Do not guess the GoogleDrive account folder name.

## 4. Start the dashboard

From the repository root:

```bash
source .venv-mac/bin/activate
streamlit run 07_dashboard/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## 5. Run AI processing in Colab

Continue using the existing Colab GPU notebook and `requirements_demo.txt`. Mount the same Drive folder in Colab and run the complete pipeline there. The Mac dashboard reads the generated meeting JSON and PDF artifacts from the synced project folder.

## Important boundaries

- This Mac environment is **dashboard-only**; do not click/run the new audio-processing workflow locally.
- Do not install the Colab `requirements_demo.txt` into the Mac virtual environment.
- Keep model caches and virtual environments outside Google Drive.
- The existing Colab requirements and pipeline logic are retained; `networkx` and `reportlab` are explicitly listed because the DAG and PDF export use them.
- No Phase 1 or Phase 2 algorithms were changed. The Mac dashboard uses the shared project-root configuration, validates its two mandatory startup JSON files, and can load a separately saved Phase 2 result. The Intel Mac setup remains display-only; AI processing stays in Colab.
