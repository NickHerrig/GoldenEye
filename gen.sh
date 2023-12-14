#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <api_key> <prompt>"
    exit 1
fi

API_KEY=$1
PROMPT=$2

for i in {1..5}
do
   curl -X POST "https://api.roboflow.com/synthetic-image?api_key=$API_KEY" \
    -H 'Content-Type: application/json' \
    --data \
    "{
      \"project_url\": \"goldeneye\",
      \"prompt\": \"$PROMPT\"
    }"
   echo "Call $i completed"
done

