#!/bin/sh
set -e

echo Starting Hallo4 RunPod H100 test worker
python /app/download_hallo4_models.py
cd /app/hallo4
echo Starting Hallo4 serverless handler
python -u /app/handler_hallo4.py
