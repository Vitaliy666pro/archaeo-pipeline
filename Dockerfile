# Use a minimal conda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy and create the 'archaeo' environment via mamba
COPY environment.yml /tmp/environment.yml
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy

# Make the new environment’s binaries available
ENV PATH /opt/conda/envs/archaeo/bin:$PATH

# Copy all project files into the container
COPY . /app

# Default command: start JupyterLab inside the archaeo env
CMD ["conda", "run", "-n", "archaeo", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
