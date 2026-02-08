# VoxCity Web App

A web-based interface for the [VoxCity](https://github.com/kunifujiwara/VoxCity) package, built with **FastAPI** (backend) and **React + Vite** (frontend).

## Prerequisites

- Python 3.9+
- Node.js 18+
- The `voxcity` package installed (from the project root: `pip install -e .`)

## Quick Start

```bash
# 1. Install backend dependencies
pip install -r app/backend/requirements.txt

# 2. Run both backend & frontend
python app/run.py
```

The script will automatically install npm dependencies on first run.

- **Backend** → http://localhost:8000
- **Frontend** → http://localhost:3000

## Manual Start

### Backend

```bash
cd app
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd app/frontend
npm install    # first time only
npm run dev
```

## Architecture

```
app/
├── run.py                   # Convenience launcher
├── backend/
│   ├── main.py              # FastAPI endpoints
│   ├── models.py            # Pydantic request/response models
│   ├── state.py             # In-memory application state
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── package.json
    ├── vite.config.ts       # Vite config (proxies /api → backend)
    ├── index.html
    └── src/
        ├── App.tsx          # Main component & tab routing
        ├── api.ts           # Typed API client
        ├── components/      # Shared components
        │   ├── MapPicker.tsx
        │   └── PlotlyViewer.tsx
        └── tabs/            # One component per tab
            ├── TargetAreaTab.tsx
            ├── GenerationTab.tsx
            ├── SolarTab.tsx
            ├── ViewTab.tsx
            ├── LandmarkTab.tsx
            └── ExportTab.tsx
```

## Features

| Tab | Description |
|-----|-------------|
| **Target Area** | Select area by drawing on map, entering coordinates, or geocoding a city |
| **Generation** | Generate VoxCity 3D model (OSM or CityGML sources) |
| **Solar** | Compute solar irradiance (instantaneous or cumulative) |
| **View** | View index analysis (green view, sky view, custom classes) |
| **Landmark** | Landmark visibility mapping |
| **Export** | Export to CityLES or OBJ formats |
