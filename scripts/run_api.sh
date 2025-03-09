#!/bin/bash

# Build the Docker image
docker build -t gbm_model_api .

# Run the container
docker run -p 8080:8080 gbm_model_api
