#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "Using project folder: $(pwd)"

if [ ! -d ".venv" ]; then
  echo "Creating local virtual environment in .venv ..."
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Remove old outputs so stale figures cannot be confused with this version.
rm -rf outputs

python run_1d_layered_reflection_demo.py

echo
echo "Done. Open outputs/layered_reflection_demo/metadata.json and outputs/layered_reflection_demo/01_velocity_model_layers.png."
