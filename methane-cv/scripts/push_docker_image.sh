#!/bin/bash
set -e
set -x  # debug mode

TAG=$1

# run this script from the root directory so docker can find the Dockerfile
docker build -t orbioazure.azurecr.io/methane-cv:"$TAG" -f methane-cv/Dockerfile .

# may need to add --identity to `az login`
# if you have a ManagedIdentity set up and want to use it
# this will direct you to open a web browser to login
az login
az acr login -n orbioazure.azurecr.io

# Image should be tagged as orbioazure.azurecr.io/methane-cv:<tag>

docker push orbioazure.azurecr.io/methane-cv:"$TAG"
