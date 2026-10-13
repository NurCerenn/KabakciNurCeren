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