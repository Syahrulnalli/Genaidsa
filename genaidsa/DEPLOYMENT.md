# Publish on Streamlit Community Cloud

This repository is prepared as a single Streamlit app. The entry point is `frontend/app.py`, and the root `requirements.txt` installs Streamlit, PyTorch, Diffusers, and the image-processing dependencies. The app calls the Python VAE pipeline directly, so deployment needs no backend URL setting, secrets, separate service, or paid cloud resources.

## Current publishing readiness

- GitHub authentication is available in the connected GitHub account (`Syahrulnalli`).
- A public, empty repository named [`Syahrulnalli/Genaidsa`](https://github.com/Syahrulnalli/Genaidsa) exists and matches this project. It has no project files yet.
- Git is not installed or available on the local PowerShell `PATH`, and this workspace does not yet have a `.git` directory. The project changes are therefore prepared locally but have not been pushed.
- No Streamlit Community Cloud deployment has been created or tested, so there is no public app URL yet.

## Push the prepared project

1. Install Git for Windows from [git-scm.com/download/win](https://git-scm.com/download/win), then reopen PowerShell. Confirm `git --version` works.
2. In PowerShell, move to this project folder and create the local repository:

   ```powershell
   Set-Location 'C:\Users\Syahrul Nalli\Pictures\alliance\genaidsa'
   git init -b main
   git add .
   git status --short
   ```

3. Review `git status --short` to confirm it contains only project files. The `.gitignore` excludes virtual environments, model weights, local settings, secrets, and caches.
4. Commit the files and push to the matching empty GitHub repository:

   ```powershell
   git commit -m "Prepare single-app Streamlit deployment"
   git remote add origin https://github.com/Syahrulnalli/Genaidsa.git
   git push -u origin main
   ```

   Git may open a browser so you can sign in to GitHub. If Git reports that the `origin` remote already exists, inspect it with `git remote -v` before changing it.

## Create the Streamlit app

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub. If asked, authorize Streamlit Community Cloud to access `Syahrulnalli/Genaidsa`.
2. Choose **Create app** and select repository `Syahrulnalli/Genaidsa`, branch `main`, and file path `frontend/app.py`.
3. In **Advanced settings**, select Python 3.11. Leave Secrets empty; the app does not need a backend URL or credentials.
4. Deploy. Allow time for package installation and the first download of the pretrained `stabilityai/sd-vae-ft-mse` model.
5. Open the assigned `*.streamlit.app` URL, upload an image and process it. Confirm that all three images and metrics appear, then download the reconstructed PNG.

Streamlit Community Cloud deploys from GitHub and assigns the app URL after a successful deployment. Until the Git push and cloud test above are complete, the project has no public app URL. See the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for the current Community Cloud flow.
