#!/bin/bash

# Check if image directory path is provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <image_directory>"
    exit 1
fi

IMG_DIR="$1" # Get the image directory path from the script argument

# Loop through each image in the directory
for img in "${IMG_DIR}"/*.jpg; do
    # Extract the base name of the image without extension
    base_name=$(basename "$img" .jpg)

    # Define the output file name based on the image name
    output_file="${IMG_DIR}${base_name}.txt"

    # Execute the command and save the output to the defined output file
    /mnt/AI/tools/llama.cpp/build/bin/llava-cli -m /mnt/AI/models/ggml-model-q5_k.gguf --mmproj /mnt/AI/models/mmproj-model-f16.gguf --temp 0.1 -p "retrive the original text content from the images which are related to Tietoevry company's HR meterials, list the text content found." --image "$img" > "$output_file"
done

