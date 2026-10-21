#CLAUDE>> Lines tagged "#CLAUDE>>" were written by Claude (an AI); lines tagged
#CLAUDE>> "#DAN>>" were written by Dan. These are learning suggestions only.
#CLAUDE>> GOOD: this twelve-line file is where the modular design pays off: the whole pipeline is
#CLAUDE>> one loop over the two datasets, and a third climate variable would be one more tuple.
#CLAUDE>> It could BE main.py, with the exploration kept in a separate script.
# my attempt to write everythig into a single loop

import read_data
import analysis

for path, label in [("UnitModularity/USAAnnualTemp1950_2008.rds", "temperature"),
                    ("UnitModularity/USAAnnualPcpn1950_2008.rds", "precipitation")]:
    df = read_data.load_data(path)
    clean = read_data.clean_data(df)
    slopes = analysis.compute_all_slopes(clean)
    print(label, "stations:", len(slopes), "mean slope:", slopes["slope"].mean())
    analysis.plot_histogram(slopes, label)
    analysis.plot_map(slopes, label)