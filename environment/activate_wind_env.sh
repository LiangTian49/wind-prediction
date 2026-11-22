#!/bin/bash
# Source this file. Batch shells may not inherit the login shell's stage search path.
module --force purge
module use /p/software/deep/otherstages
module load Stages/2025
module load GCCcore/.13.3.0 Python/3.12.3
source /p/project1/sdlrs/tian1/wind_prediction/wind-env/bin/activate
export PYTHONNOUSERSITE=1
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "wind-env needs Python 3.12"; assert sys.prefix == "/p/project1/sdlrs/tian1/wind_prediction/wind-env", "wrong virtual environment"'
