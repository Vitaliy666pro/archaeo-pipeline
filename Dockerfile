 FROM continuumio/miniconda3
 WORKDIR /app

 COPY environment.yml /tmp/environment.yml
 RUN conda install -y -n base -c conda-forge mamba && \
     mamba env create -f /tmp/environment.yml && \
     conda clean -afy

 ENV PATH="/opt/conda/envs/archaeo/bin:/opt/conda/bin:${PATH}"

 COPY . /app

ENTRYPOINT ["/opt/conda/envs/archaeo/bin/jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
