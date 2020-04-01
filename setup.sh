#/bin/bash

python setup.py develop
cd data
python generate_data.py --fetch --output-population ../assets/population.json --output-cases ../assets/case_counts.json

