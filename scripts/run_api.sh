#!/bin/bash

# Build the Docker image
docker build -t GBM_model_api .

# Run the container
docker run -p 8080:8080 GBM_model_api
