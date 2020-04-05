# pydemic

Clone this repo with `git clone --recurse-submodules ...`

Install with `sh setup.sh`, which does the following:
* installs `pydemic` via `python setup.py develop`, which will install all dependencies
* fetches and parses data via `python data/parse_all.py --fetch --output-population assets --output-cases assets`

`json` datafiles are stored in `assets`.
