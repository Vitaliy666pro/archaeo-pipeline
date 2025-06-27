# Archaeo-Pipeline

A containerized data pipeline for archaeological site analysis, featuring raw data ingestion, processing, model training, and geospatial visualizations.

## Repository Structure

```
archaeo-pipeline/
├── .env.example             # Template for runtime secrets
├── environment.yml          # Conda/Mamba environment specification
├── Dockerfile               # Defines the Docker image and JupyterLab entrypoint
├── docker-compose.yml       # Service definition, volume mounts, and ports
├── Makefile                 # Convenience shortcuts (download-data, build, up)
├── README.md                # Project overview and Quick Start instructions
├── setup_instructions.py    # Quick start instructions (also available as PDF)
├── setup_instructions.pdf   # Quick start instructions in PDF format
├── config.yaml              # (Optional) Static paths and model parameters
├── scripts/
│   └── download_data.sh     # Script to fetch raw datasets into `data/`
├── data/
│   ├── raw/                 # Raw downloaded data (not in VCS)
│   └── interim/             # Processed data outputs (not in VCS)
├── notebooks/
│   └── legacy/              # Original Jupyter drafts (read-only)
├── results/
│   ├── maps/                # Generated PNGs, heatmaps, geojsons
│   ├── report.pdf           # Final PDF report
│   ├── final_shortlist.csv  # Top candidates
│   └── candidates_2000.geojson
├── src/                     # Refactored modules
│   ├── data_io.py           # Data loading & cleaning functions
│   ├── features.py          # Feature extraction logic
│   ├── train.py             # Model training pipeline
│   ├── predict.py           # Batch prediction script
│   ├── viz.py               # Map and figure generation
│   └── config.py            # Config loader for `config.yaml`
└── tests/                   # (Optional) Unit and integration tests
    └── test_data_io.py
```

## Quick Start Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/archaeo-pipeline.git
   cd archaeo-pipeline
   ```

2. **Configure runtime secrets**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your keys:

   ```bash
   JUPYTER_TOKEN=my-super-secret-token
   OPENAI_API_KEY=your-openai-api-key
   OPENTOPO_KEY=your-opentopo-key
   COPERNICUS_CLIENT_ID=your-copernicus-client-id
   COPERNICUS_CLIENT_SECRET=your-copernicus-client-secret
   ```

3. **(Optional) Download raw data**

   ```bash
   make download-data
   # or
   bash scripts/download_data.sh
   ```

4. **Build the Docker image**

   ```bash
   docker-compose build
   ```

5. **Start JupyterLab**

   ```bash
   docker-compose up -d
   ```

6. **Open JupyterLab**
   Navigate to:

   ```
   http://<VM_IP_or_localhost>:8888/?token=my-super-secret-token
   ```

7. **Stop the environment**

   ```bash
   docker-compose down
   ```
