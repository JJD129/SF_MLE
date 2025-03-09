#!/bin/bash

# Build the Docker image
docker build -t my_model_api .

# Run the container
docker run -p 8080:8080 my_model_api
