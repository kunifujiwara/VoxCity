# syntax=docker/dockerfile:1.6

FROM node:20-bookworm-slim AS frontend

WORKDIR /work

COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci

COPY app/frontend/ ./
RUN npm run build

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        gfortran \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libgl1 \
        libglib2.0-0 \
        liblapack-dev \
        libopenblas-dev \
        libproj-dev \
        libspatialindex-dev \
        proj-bin \
        proj-data \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        wget \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Install the voxcity library from this repo's own source (this repo IS voxcity).
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && printf 'numpy>=1.24,<2\nnumba<0.60\nxarray<2025.1\nrasterio==1.3.11\nfiona<1.10\n' > /tmp/constraints.txt \
    && python -m pip install -c /tmp/constraints.txt numpy \
    && python -m pip install -c /tmp/constraints.txt GDAL==$(gdal-config --version) --no-build-isolation \
    && python -m pip install -c /tmp/constraints.txt ".[gpu]"

# App code + backend runtime deps (fastapi/uvicorn/python-multipart/python-dotenv).
COPY app ./app
RUN python -m pip install -c /tmp/constraints.txt -r app/backend/requirements.txt

# Built frontend from the node stage → served by the backend SPA route.
COPY --from=frontend /work/dist ./app/frontend/dist

# Container defaults (override at run time via docker-compose / .env.docker).
#   VOXCITY_DATA_DIR / VOXCITY_OUTPUT_DIR  base data + output roots (bind-mounted).
#   VOXCITY_SESSION_MAX_UPLOAD_MB          max session upload size in MB (backend cap).
# CITYGML_PATH is intentionally NOT baked here; supply it via docker-compose so the
# image is not tied to a specific PLATEAU dataset name.
# GEE_PROJECT (Earth Engine) is intentionally NOT baked; supply it per-deployment.
ENV VOXCITY_DATA_DIR="/work-voxcity/data" \
    VOXCITY_OUTPUT_DIR="/work-voxcity/output" \
    VOXCITY_SESSION_MAX_UPLOAD_MB="500" \
    DISPLAY=":99"

RUN mkdir -p /work-voxcity/data /work-voxcity/output

EXPOSE 8000

# Match app/run.py: uvicorn target `backend.main:app` with the app/ dir as CWD.
WORKDIR /app/app
ENTRYPOINT ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
