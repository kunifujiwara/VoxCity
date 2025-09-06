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
    gdal-bin libgdal-dev \
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

# Set up Python environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install GDAL Python bindings (version should match system GDAL)
RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

# Install geospatial Python packages
RUN pip install \
    numpy \
    pandas \
    geopandas \
    shapely \
    fiona \
    pyproj \
    rasterio \
    earthengine-api \
    geemap

# Install VoxCity and its dependencies
RUN pip install voxcity

# Install web app specific packages
RUN pip install \
    streamlit>=1.28.0 \
    streamlit-folium>=0.15.0 \
    folium>=0.14.0 \
    matplotlib>=3.7.0 \
    plotly>=5.17.0 \
    pillow \
    scipy \
    scikit-image

# Install additional packages that might be needed for VoxCity
RUN pip install \
    trimesh \
    pyvista \
    vtk \
    open3d

# Create a non-root user FIRST
RUN useradd -m -s /bin/bash voxcity

# Create workspace directory
WORKDIR /app

# Copy the web application files
COPY app.py /app/
COPY requirements.txt /app/

# Create necessary directories for outputs with proper permissions
RUN mkdir -p /app/output/test \
    /app/output/envimet \
    /app/output/magicavoxel \
    /app/output/obj \
    /app/output/solar \
    /app/output/view \
    /app/output/3d_vis && \
    chown -R voxcity:voxcity /app && \
    chmod -R 755 /app/output

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
streamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Default command
CMD ["/app/entrypoint.sh"]