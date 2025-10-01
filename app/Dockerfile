# Dockerfile (updated)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    git python3 python3-pip python3-dev python3-venv \
    wget curl vim nano htop \
    build-essential cmake \
    # Geospatial libraries
    gdal-bin libgdal-dev python3-gdal \
    libspatialindex-dev \
    libproj-dev proj-data proj-bin \
    libgeos-dev \
    # Graphics and visualization libraries (for VoxCity 3D rendering)
    libgl1-mesa-glx libglu1-mesa \
    xvfb \
    # Additional libraries that might be needed
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment (allow access to system site-packages for python3-gdal)
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Constrain scientific stack to avoid NumPy 2 ABI issues
RUN printf "numpy<2\nnumba<0.60\nxarray<2025.1\nrasterio==1.3.11\nfiona<1.10\n" > /tmp/constraints.txt

# Install VoxCity and app dependencies under constraints
RUN pip install -c /tmp/constraints.txt \
    voxcity \
    streamlit>=1.28.0 \
    streamlit-folium>=0.15.0 \
    folium>=0.14.0 \
    matplotlib>=3.7.0 \
    plotly>=5.17.0 \
    pillow \
    scipy \
    scikit-image \
    requests \
    geopy \
    seaborn \
    contextily \
    tqdm \
    ipython \
    pandas \
    geopandas \
    shapely \
    pyproj \
    earthengine-api \
    geemap \
    trimesh \
    pyvista \
    vtk \
    open3d

# Create a non-root user FIRST
RUN useradd -m -s /bin/bash voxcity

# Create workspace directory
WORKDIR /app

# Copy the web application file
COPY app.py /app/

# Prepare runtime output directories used by the app (/tmp/voxcity_output)
RUN mkdir -p /tmp/voxcity_output/test \
    /tmp/voxcity_output/envimet \
    /tmp/voxcity_output/magicavoxel \
    /tmp/voxcity_output/obj \
    /tmp/voxcity_output/solar \
    /tmp/voxcity_output/view \
    /tmp/voxcity_output/epw \
    /tmp/voxcity_output/netcdf \
    /tmp/voxcity_output/3d_vis && \
    chown -R voxcity:voxcity /tmp/voxcity_output && \
    chmod -R 777 /tmp/voxcity_output

# Switch to non-root user
USER voxcity

# Set environment variables for Earth Engine
ENV EARTHENGINE_CREDENTIALS=/home/voxcity/.config/earthengine/credentials

# Expose Streamlit default port
EXPOSE 8501

# Set up virtual display for headless 3D rendering
ENV DISPLAY=:99

# Create an entrypoint script as voxcity user
RUN echo '#!/bin/bash\n\
# Start virtual display\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
# Wait for display to be ready\n\
sleep 2\n\
# Run streamlit app\n\
/opt/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Default command
CMD ["/app/entrypoint.sh"]