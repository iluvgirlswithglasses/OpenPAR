#!/bin/bash
# upload one system from OpenPAR to Google Drive
# e.g. `./colab/upload.sh UniPAR`

# move to correct directory
SCRIPT_DIR="$(dirname $(realpath "$0"))"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# upload $1 to Google Drive
zip -r ./$1.zip ./$1
rclone copy ./$1.zip gdrive:/yomikawa-reid/src/
rm ./$1.zip
