#!/bin/bash
set -e
echo 'Building ISLES 2026 submission Docker image...'
docker build -t isles26-submission .
echo 'Exporting container archive for Grand Challenge...'
docker save isles26-submission | gzip -c > isles26_submission.tar.gz
echo 'Done! Output: isles26_submission.tar.gz'
