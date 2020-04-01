# pydemic

Clone this repo with `git clone --recurse-submodules ...`

Install with `sh setup.sh`, which does the following:
* installs `pydemic` via `python setup.py develop`, which will install all dependencies
* fetches and parses data via `python data/parse_all.py --fetch --output-population assets --output-cases assets`

`json` datafiles are stored in `assets`.


Quick and dirty result/data save script:
```python


def save_data(result, filename):
    from datetime import datetime, timedelta
    dates = [ datetime(2020,1,1)+timedelta(days=x) for x in result.t ]
    compartments = {}
    fp = open(filename, 'w')
    fp.write("time\t")
    for compartment in result.compartments:
        compartments[compartment] = result.y[compartment].sum(axis=-1)
        fp.write(compartment + "\t")
    fp.write("\n")
    for i in range(len(dates)):
        fp.write(dates[i].strftime("%y-%m-%d")+"\t")
        for compartment in compartments:
            fp.write("{0:g}\t".format(compartments[compartment][i]))
        fp.write("\n")
    fp.close()
```
