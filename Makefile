.PHONY: download-data build up

download-data:
	# download raw data into data/raw
	bash scripts/download_data.sh

build:
	# build docker image using mamba-based Dockerfile
	docker-compose build

up:
	# start JupyterLab service
	docker-compose up --build
