#!/bin/bash
# Run from the root Git directory: ./init.sh

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
