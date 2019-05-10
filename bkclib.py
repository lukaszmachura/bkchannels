import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


class MultistablePDF(UserWarning):
    pass


def read_atf(filename, skiprows=10):
    """
    Reads atf file with BK channel signal.
    Returns Pandas DataFrame.
    """

    data = pd.read_csv(filename, skiprows=skiprows, delimiter="\t", names=["t", "C"],
                       dtype={"t": "float64", "C": "float64"})

    return data


def recalculate_bins(b):
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    return (b[:-1] + b[1:]) / 2


def histogram(data, bins=133, density=True):
    h, b = np.histogram(data, bins=bins, density=density)
    return h, recalculate_bins(b)


def locate_stages(pdf, method='middlepoint'):
    if method == 'middlepoint':
        return locate_stages_middlepoint(pdf)
    else:
        raise NotImplementedError("method not implemented")


def locate_stages_middlepoint(hist, bins, extrema, poly=9, dict=False):
    while len(bins) > len(hist):
        bins = recalculate_bins(bins)

    Lidx, Midx, Ridx = extrema
    L, M, R = bins[Lidx], bins[Midx], bins[Ridx]

    p = np.polyfit(bins[Lidx:Ridx+1], hist[Lidx:Ridx+1], poly)
    deriv = np.polyder(p)

    tangent = []
    for point in [0.5 * (L + M), M, 0.5 * (M + R)]:
        w = np.polyval(p, point)
        a = np.polyval(deriv, point)
        b = -a * point + w
        tangent.append([a, b])
    Lt, Mt, Rt = tangent

    A = (Mt[1] - Lt[1]) / (Lt[0] - Mt[0])
    B = (Mt[1] - Rt[1]) / (Rt[0] - Mt[0])

    if dict:
        return {'A': A, 'B': B,
                'fit': p,
                'tangents': tangent}

    return A, B


def locate_maxima(hist, bins):
    if not isinstance(hist, np.ndarray):
        hist = np.array(hist)

    hmaxidx = np.where(hist == max(hist))[0][0]
    hprim = np.diff(hist)
    bprim = recalculate_bins(bins)

    ret = []
    for idx in range(hmaxidx - 2, hprim.size - 2):
        if hprim[idx] * hprim[idx+1] < 0 and\
           hprim[idx-1] * hprim[idx+2] < 0 and\
           hprim[idx-1] * hprim[idx] > 0 and\
           hprim[idx+1] * hprim[idx+2] > 0:
                # ret.append(np.mean(bprim[idx-1:idx+3]))
                i = idx + 1
                ret.append((hist[i], i))

    if len(ret) > 3:
        warnings.warn("Possible multistable pdf, returning last 3", MultistablePDF)

    return ret


def locate_best_maxima(h, b):
    extrema = locate_maxima(h, b)
    extrema.sort(reverse=True)
    return sorted(list(list(zip(*extrema[:3]))[1]))


def get_A_and_B(datafile, pm='minus', bins=333):
    if "clean" not in datafile:
        data = read_atf(datafile)
        data = prepare_data(data)
    else:
        data = read_atf(datafile, skiprows=1)

    h, b = histogram(data, bins=bins)
    h = moving_average(h, spencer_filter())
    lspencer = len(spencer_filter()) // 2
    b = b[lspencer:-lspencer]
    Lidx, Midx, Ridx = locate_best_maxima(h, b)
    lsm = locate_stages_middlepoint(h, b, (Lidx, Midx, Ridx), 9, True)

    A, B = lsm['A'], lsm['B']
    return A, B


def run_check(fname, skiprows=0, bins=333):
    # data = read_atf("sample.atf")
    # data = read_atf("data/2019_04_04_00013_20D.atf")
    data = read_atf(fname, skiprows=skiprows)
    name = fname[:-4]

    # bins = bins #int(np.sqrt(len(data))) #25
    data = data["C"]
    mean = data.median()

    h, b = histogram(data, bins=bins)
    plt.plot(b, h, '-x')

    # replace h with moving moving_average
    h = moving_average(h, spencer_filter())
    lspencer = len(spencer_filter()) // 2
    b = b[lspencer:-lspencer]

    # plt.plot(b, h, '-o')
    # plt.savefig('tmp.png')
    # plt.close()

    hprim = np.diff(h)
    bprim = recalculate_bins(b)
    Lidx, Midx, Ridx = locate_best_maxima(h, b)
    lsm = locate_stages_middlepoint(h, b, (Lidx, Midx, Ridx), 9, True)
    A, B = lsm['A'], lsm['B']

    # historgam plot
    ax = data.hist(bins=bins, normed=True, log='y')
    fig = ax.get_figure()
    # another historgam plot
    plt.plot(b, h, '-')
    # extrema location
    for idx in [Lidx, Midx, Ridx]:
        plt.plot([b[idx]], [np.polyval(lsm['fit'], b[idx])], 'or')
    # polyfit plot
    for x in [A, B]:
        plt.plot([x], [np.polyval(lsm['fit'], x)], 'ok')
    fig.savefig(name + '_hist.png')
    plt.close()

    plt.plot(bprim, hprim, 'o-')
    plt.axhline(0, color='k')
    # plt.plot([b[Lidx], b[Midx], b[Ridx]], [0] * 3, 'or')
    plt.savefig(name + '_histprim.png')


def run_sample_check():
    run_check('sample.atf')


def is_close(a, b, eps=0.1):
    return abs(1 - a / b) < eps
    # return abs(a - b) < eps


def detect_drop(data, window=0.1, step=0.001):
    ld = len(data)
    window = int(ld * window)
    step = int(ld * step)
    for idx in range(0, ld - window - step, step):
        a = np.max(data[idx:idx+window]) - np.min(data[idx:idx+window])
        b = np.max(data[idx+step:idx+window+step]) - np.min(data[idx+step:idx+window+step])
        print(idx, a, b, is_close(a, b, 0.3))
        if not is_close(a, b):
            return idx
    return ld


def plot_data(fname, maks=-1):
    data = read_atf(fname)
    name = fname[:-4]
    data = data["C"][data["C"] < 0]
    ax = data[0:maks:1000].plot()
    fig = ax.get_figure()
    fig.savefig(name + '_current.png')
    plt.close()

def clean_data():
    data_clean_db ={
    "data/2019_04_04_00013_20D.atf": {'idx': [(3397266, -1)], 'C': (-30, 0)},
    "data/2019_04_04_00011_40D.atf": {'idx': [(318600, 348434), (939017, 1033931), (1850000, -1)], 'C': (-40, -5)},
    "data/2019_04_04_0003_50D.atf": {'idx': [(1674875, 2363730), (2954000, -1)], 'C': (-30, 0)},
    "data/2019_04_04_00015_60D.atf": {'idx': [(2094730, -1)], 'C': (-50, -10)},
    # "data/2019_04_04_0009_80D.atf": (),
    }

    for fname in data_clean_db:
        data = read_atf(fname)
        name = fname[:-4]

        # remove data chunks with some suspicious C
        details = data_clean_db[fname]
        for od, do in details['idx']:
            data.drop(data.index[od:do], inplace=True)

        # ...and higher/lower then selected C
        Cod, Cdo = details['C']
        data = data[(data["C"] < Cdo) & (data["C"] > Cod)]

        # write to file similar to original ATF
        with open(name + '_clean.atf', 'w') as f:
            data.to_csv(f, index=False, sep='\t')


def prepare_data(fname, od=0, do=-1, negative=True):
    data = read_atf(fname)
    if negative:
        data = data[data["C"] < 0]
    else:
        data = data[data["C"] > 0]
    return data

    # name = fname[:-4]
    # with open(name + '_clean.atf', 'w') as f:
    #     data[od:do].to_csv(f, index=False, sep='\t')


def spencer_filter():
    return [i / 320 for i in (-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 21, 3, -5, -6, -3)]


def moving_average(data, filter=None):
    """Smoothing the data with moving average.
    IN:
    data: list, numpy array or DataSeries
    filter: INT for uniform filter
            SEQUENCE for filter
            None (default) 3 point uniform spencer_filter
    OUT:
    numpy array with filtered signal
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if filter == None:
        filter = np.array([1/3, 1/3, 1/3])

    if isinstance(filter, int):
        filter = np.array([1 / filter for i in range(filter)])
    elif isinstance(filter, (list, tuple)):
        filter = np.array(filter)
    else:
        raise ValueError("None, INT, LIST or TUPLE")

    return np.convolve(data, filter, 'valid')


if __name__ == "__main__":

    # clean_data()
    # prepare_data(fname)
    # data = read_atf(fname)
    # data = data["C"][data["C"] < 0]
    # dd = detect_drop(data)
    # print(dd, len(data))
    # plot_data(fname, 3500000)
    # plot_data("data/2019_04_04_00013_20D.atf")
    # run_check('data/2019_04_04_00013_20D_clean.atf', skiprows=1)
    # print(run_sample_check())
    import glob
    for fn in glob.glob('data/*_clean.atf'):
        try:
            print(fn)
            plot_data(fn)
            run_check(fn, skiprows=1)
        except Exception as inst:
            print (type(inst))     # the exception instance
            print (inst.args)      # arguments stored in .args
            # print (inst)
            print(fn, "...lipa")
