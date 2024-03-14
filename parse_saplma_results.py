"""
Run on output of TrainProbes.py, e.g. `python parse_saplma_results.py saplma_results.log`
to give a nice table of results
"""
import sys
import pandas as pd


def get_table_from_script_output(log_file):
    rows = []
    with open(log_file) as file:
        for line in file.readlines():
            if not line.startswith('dataset: '):
                continue
            line = line.replace(': ', ':')
            rows.append(dict([tuple(entry.split(':')) for entry in line.split()]))

    return pd.DataFrame(rows)


if __name__ == '__main__':
    log_file = sys.argv[1]
    df = get_table_from_script_output(log_file)
    print(df)
