#!/bin/bash
# Run from the root Git directory: ./init.sh

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  ENV_FILE=conda_env_linux.yml
elif [[ "$OSTYPE" == "darwin"* ]]; then
  ENV_FILE=conda_env_macos.yml
else
  echo -e Unsupported OS type: \""$OSTYPE"\"! This software is compatible with systems from \"linux-gnu\" or \"darwin\" families.
  exit 1
fi

# Clear conda package cache to avoid version conflict
echo -e "Clearing conda package cache...\n"
conda clean -y --packages --tarballs

# Create conda environment and install modules listed in conda-env.yml
echo -e "\nCreating a new conda environment at .conda_env...\n"
conda env create -f "$ENV_FILE" -p .conda_env
