# Archaeo-Pipeline

A Dockerized end-to-end data pipeline for archaeological site detection and verification.

---

## Repository Layout

```
archaeo-pipeline/
├── .env.example             # Template for runtime secrets
├── .env                     # Your local secrets (not in VCS)
├── secrets/                 # Service‐account JSONs (not in VCS)
│   └── gg-sa-key.json       # GCP Earth Engine service account key
├── environment.yml          # Conda/Mamba environment specification
├── Dockerfile               # Defines the Docker image and JupyterLab entrypoint
├── docker-compose.yml       # Service definition, volume mounts, and ports
├── Makefile                 # Convenience shortcuts (download-data, build, up, down)
├── config.yaml              # Static paths, CRS, and model parameters
├── main.py                  # Script to run full pipeline end-to-end
├── datasets_metadata.json   # Datasets with description and links
├── scripts/
│   ├── data_preprocessing/
│   │   ├── unpack_data.py
│   │   ├── combine_datasets.py
│   │   └── download_external_datasets.py
│   ├── feature_engine/
│   │   ├── get_tiles_with_regions_and_sites.py
│   │   ├── get_rivers_and_mountains.py
│   │   ├── get_soil_features.py
│   │   ├── get_emb_pca.py
│   │   └── cut_roads.py
│   ├── model/
│   │   └── train_model.py
│   └── verification/
│       └── download_predicted_s1_s2.py
├── data/
│   ├── raw/                 # Raw downloaded data (not in VCS)
│   └── interim/             # Intermediate outputs (not in VCS)
│   └── processed/
│   └── datasets.7z 
├── results/
│   ├── predicted/           # S1/S2 & LiDAR hillshade downloads
│   ├── maps/                
│   └── candidates_top500.csv # Final shortlist
└── notebooks/               # Analysis & demos
```

---

## `.env` Keys

Copy and fill in **exactly** these variables in your project root:

```ini
OPENTOPO_KEY=YOUR_OPENTOPO_KEY
OPENAI_API_KEY=YOUR_CLASSIC_SK_KEY
JUPYTER_TOKEN=YOUR_JUPYTER_TOKEN
EE_INIT_PROJ=YOUR_EE_PROJECT_ID
GSA_EMAIL=your-sa@your-project.iam.gserviceaccount.com
```

> **Note:**
> - Do **not** commit `.env` or `secrets/` contents to VCS.
> - Place your Earth Engine–enabled GCP service-account JSON at `secrets/gg-sa-key.json`.

---

## Quick Start

1. **Clone & enter**  
   ```bash
   git clone https://github.com/your-org/archaeo-pipeline.git
   cd archaeo-pipeline
   ```

2. **Configure `.env`**  
   ```bash
   cp .env.example .env
   # then edit .env to set your keys as shown above
   ```

3. **Add GCP JSON key**  
   ```text
   secrets/gg-sa-key.json
   ```

4. **Build & launch**  
   ```bash
   docker-compose build
   docker-compose up -d
   ```
   Open JupyterLab at:  
   `http://localhost:8888/?token=<JUPYTER_TOKEN>`

6. **Run the pipeline**  
   ```
   select the kernel archeo_pipeline run main.py

   ```

7. **Inspect results**  
   Check `results/predicted` folder for short_list and top_500_candidates.
   Check `notebooks/` stg_3_1_visual_gpt_verification to see the promts and visual_gpt 4.1 output