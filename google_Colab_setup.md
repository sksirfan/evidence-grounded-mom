**Step 1** : Download code from GitHub

---

**Step 2** : Create a Folder named MTechIndProj/MoM_Project/
in the Google Drive ,

---

**Step 3** : Upload all the downloaded files to above Created folder in the Drive.

---

**Step 4** : Make sure to upload 3 startup files shared to its respective folders as shown below for refrences,

keep these files in drive Required for startup Load.

MoM_Project/data/transcripts/ES2004a_structured_mom_claims.json
MoM_Project/outputs/verified/ES2004a_M9_full_multi_attribute_verification.json
MoM_Project/outputs/pdf/ES2005a.Mix-Headset_Verifiable_Meeting_Minutes.pdf

---
**Step 5** : Open the Startup Python Notebook-run_mom_project.ipynb

---
**Step 6** : Connect to GPU Runtime (Recommended T4 GPU)

---
**Step 6** : Configure your Hugging Face token
This project uses Pyannote and Hugging Face models. You need a Hugging Face access token with access to the required gated models.
1. Open Hugging Face access tokens.
2. Create or copy your access token.
3. In Colab, open the Secrets panel (key icon).
4. Add a secret named HF_TOKENnew.
5. Enable notebook access to that secret.
**Never commit the Hugging Face token to GitHub.**
Do not place tokens inside source files, notebooks, README files, or committed configuration files.

---
**Step 7** : Run the Notebook (Recommended cell by cell execution)

---
**Step 8** : Load the URl provided from the CloudFare Tunnel.

Something similar to-
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://cradle-define-corpus-investment.trycloudflare.com   |

Step 9 : Streamlit Dashboard Starts ,Preprocessed Meeting ES2004a details will be loaded.
Step 10 : Upload a new meeting to Process and find Dag Details and MOM.
