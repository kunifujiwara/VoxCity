# Set up Google Earth Engine

Many of VoxCity's data sources (land cover, canopy height, and several DEMs) are
served through [Google Earth Engine](https://earthengine.google.com/). This
guide walks through the one-time setup so those sources work.

## 1. Create an Earth Engine–enabled Cloud Project

1. Sign in with a Google account that has Earth Engine access.
2. Follow Google's
   [Cloud project setup guide](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup)
   to create (or register) a Cloud Project for Earth Engine.
3. Note the **project ID** — you may need it when initializing Earth Engine.

## 2. Authenticate

### Local environment

```bash
earthengine authenticate
```

This opens a browser, asks you to grant access, and stores a token on your
machine for future sessions.

### Google Colab

```text
# Click the displayed link, generate a token, then paste it back
!earthengine authenticate --auth_mode=notebook
```

## 3. Initialize in Python

VoxCity initializes Earth Engine for you when needed, but you can initialize it
explicitly to confirm the setup and select your project:

```python
import ee

ee.Initialize(project="your-earthengine-project-id")
print("Earth Engine ready")
```

If initialization fails, re-run the authentication step above and confirm that
your Cloud Project has the Earth Engine API enabled.

## Next steps

- Choose which datasets to pull with the {doc}`data sources guide <data_sources>`.
- Return to {doc}`installation <../installation>` for the full dependency setup.
