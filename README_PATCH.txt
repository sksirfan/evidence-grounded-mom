INTEL MAC GITHUB UPDATE PATCH

Files and destinations:
- config.py                         -> repository root (replace)
- 07_dashboard/app.py              -> repository 07_dashboard/ (replace)
- 07_dashboard/requirements_demo.txt -> repository 07_dashboard/ (replace; Colab GPU requirements)
- requirements_mac.txt             -> repository root (add)
- environment_check_mac.py         -> repository root (add)
- MAC_SETUP.md                     -> repository root (add/replace)

This update keeps the Intel Mac dashboard lightweight. It does not install or
run WhisperX, PyTorch, pyannote, or the AI processing pipeline locally. The
AI pipeline remains in Google Colab GPU. Meeting data stays in Google Drive.

Before committing, keep a local backup of the existing app.py and config.py.
Do not upload meeting JSON, recordings, transcripts, model weights, or generated
outputs to the public GitHub repository.
