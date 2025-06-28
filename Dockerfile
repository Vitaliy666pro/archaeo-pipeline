FROM continuumio/miniconda3
WORKDIR /app

COPY environment.yml /tmp/environment.yml
RUN conda install -y -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/archaeo/bin:/opt/conda/bin:${PATH}"

# Register the archaeo kernel inside the environment
RUN python -m ipykernel install \
    --sys-prefix \
    --name archaeo \
    --display-name "Python (Archaeo-Pipeline)"

RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install --skip-repo

COPY . /app

ENTRYPOINT ["/opt/conda/envs/archaeo/bin/jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
