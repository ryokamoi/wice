conda env create --name wice_public --file environment.yml --prune
conda activate wice_public

export PYTHONPATH=$PYTHONPATH:./
