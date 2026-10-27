# Online deployment preparation

The application is prepared for Streamlit Community Cloud, but it does not have a public deployment yet. The Streamlit frontend needs a public HTTPS Azure Functions backend URL; `localhost:7071` only works on the development computer.

## Frontend configuration prepared

- Community Cloud starts the repository from its root. `frontend/app.py` uses no relative file paths, and `.streamlit/config.toml` is at the required root location.
- The root `requirements.txt` declares the frontend-only packages Community Cloud needs. The backend's heavier PyTorch/Diffusers requirements remain in `backend/requirements.txt`.
- Select Python 3.11 in the Community Cloud app's Advanced settings. Streamlit Community Cloud uses GitHub repositories as the app source and allows secrets to be entered in the app's deployment settings.
- `BACKEND_URL` is read from the process environment first, then `st.secrets`, then defaults to `http://localhost:7071/api/process-image` for local development.

When the real Azure endpoint exists, enter this in Community Cloud's Secrets field, replacing the placeholder with the actual deployed HTTPS endpoint:

```toml
BACKEND_URL = "https://<actual-function-app-host>/api/process-image"
```

Do not commit that value in a source file or commit `.streamlit/secrets.toml`.

## Backend deployment preparation

`backend/` is the Azure Functions project root: it contains `host.json`, the Python v2 `function_app.py`, and its own `requirements.txt`. `backend/.funcignore` excludes the local virtual environment, tests, local settings and caches from a Functions package.

After an Azure Function App exists, log in with Azure CLI and publish from the backend folder using the app's actual name:

```powershell
az login
Set-Location backend
func azure functionapp publish <actual-function-app-name>
```

Use a Python 3.11 Linux Function App and verify that its hosting plan has enough memory and execution time for PyTorch and the pretrained VAE. The first backend request downloads model weights unless the platform cache already contains them. This can require substantial download time and storage. No Azure resource has been created by this project setup.

## Connect and verify both services

1. Push this project to a GitHub repository that you control. Keep `.venv/`, `backend/local.settings.json`, `.env`, model weights and tokens out of the repository.
2. Deploy the Azure Functions backend from `backend/`, then test `POST https://<actual-function-app-host>/api/process-image` with a real image and confirm that the response includes the three Base64 PNG images, actual metrics and metadata.
3. Create the Streamlit Community Cloud app from that GitHub repository with entrypoint `frontend/app.py`, Python 3.11, and the `BACKEND_URL` secret above.
4. Upload an image to the public Streamlit app, process it, confirm that the returned metrics are present, and download the reconstructed PNG.

The frontend calls the endpoint using Python `requests` on the Streamlit server. This is a server-to-server request, so browser CORS does not apply. If the frontend is later changed to make direct browser `fetch` calls, configure the Azure Function App's allowed origins using the exact deployed Streamlit URL. Do not use a wildcard origin.

## Current account/tool blocker

At the last inspection, this workspace had no `.git` repository, and the local terminal could not find `git`, `az` or `gh`. The connected GitHub account can be read through the GitHub connector, but there is no matching project repository and that connector does not provide a repository-creation operation. There is no Azure CLI here to check Azure login status or whether an Azure Function App already exists. Therefore neither service has been deployed or assigned a public URL.

To continue with the least manual work: install Git and Azure CLI, sign in to GitHub and Azure, create a new GitHub repository and an Azure Function App with its required storage account, then publish the backend and deploy the Community Cloud app as above. Resource creation can incur charges; review Azure's current pricing and your subscription before creating resources. No cloud resource has been created or billed by these changes.
