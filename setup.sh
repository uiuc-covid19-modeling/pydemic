#/bin/bash

python setup.py develop
python data/parse_all.py --fetch --output-population assets --output-cases assets
