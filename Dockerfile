FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy and create the 'archaeo' environment via mamba
COPY environment.yml /tmp/environment.yml
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy

# Copy all project files into the image
COPY . /app

# Ensure environment binaries and conda are on PATH
ENV PATH "/opt/conda/envs/archaeo/bin:/opt/conda/bin:$PATH"

# Default entrypoint: run JupyterLab via conda-run in archaeo env
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "archaeo", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
