#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <api_key> <prompt> <image_id>"
    exit 1
fi

API_KEY=$1
PROMPT=$2
IMAGE_ID=$3

for i in {1..10}
do
   curl -X POST "https://api.roboflow.com/synthetic-image?api_key=$API_KEY" \
    -H 'Content-Type: application/json' \
    --data \
    "{
      \"project_url\": \"goldeneye\",
      \"prompt\": \"$PROMPT\",
      \"image\": \"$IMAGE_ID\",
      \"image_type\": \"id\"
    }"
   echo "Call $i completed"
done
