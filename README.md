# Quick Start Instructions for Archaeo-Pipeline

## Prerequisites
- Git installed locally  
- Docker & Docker Compose installed  

## Steps

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/archaeo-pipeline.git
   cd archaeo-pipeline

2. **Configure runtime secrets**
    Copy the example environment file:
    cp .env.example .env

    Then edit .env and set:

    JUPYTER_TOKEN=my-super-secret-token
    OPENAI_API_KEY=your-openai-api-key
    OPENTOPO_KEY=your-opentopo-key
    COPERNICUS_CLIENT_ID=your-copernicus-client-id
    COPERNICUS_CLIENT_SECRET=your-copernicus-client-secret

3. **(Optional) Download raw data**
    make download-data
    # or
    bash scripts/download_data.sh

4. **docker-compose build and up**
    docker-compose build
    docker-compose up -d

5. **Start JupyterLab**
    http://<VM_IP_or_localhost>:8888/?token=my-super-secret-token

6. **Stop the environment**
    docker-compose down
    








