import numpy as np
import pandas as pd

def read_atf(filename, skiprows=10):
    """
    Reads atf file with BK channel signal.
    Returns Pandas DataFrame.
    """

    data = pd.read_csv(filename, skiprows=skiprows, delimiter="\t", names=["t", "C"],
                       dtype={"t": "float64", "C": "float64"})

    return data


def recalculate_bins(b):
    ret = []
    for idx in range(len(b) - 1):
        ret.append((b[idx] + b[idx]) * 0.5)
    return np.array(ret)


def histogram(data, bins=133, normed=True):
    h, b = np.histogram(data, bins=bins, normed=normed)
    return h, recalculate_bins(b)


def run_sample_check():
    data = read_atf("sample.atf")
    # print(data.head())
    data = data["C"][data["C"] < 0]
    ax = data.hist(bins=133, normed=True, log='y')
    fig = ax.get_figure()
    fig.savefig('sample.png')
    return histogram(data)


if __name__ == "__main__":
    print(run_sample_check())
