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
    if len(bins) > len(hist):
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
                ret.append(idx + 1)

    if len(ret) > 3:
        warnings.warn("Possible multistable pdf, returning last 3", MultistablePDF)
    return ret[-3:]


def get_A_and_B(datafile, pm='minus', bins=33):
    data = read_atf(datafile)

    if pm == 'minus':
        data = data["C"][data["C"] < 0]
    else:
        data = data["C"][data["C"] > 0]

    mean = data.median()
    h, b = histogram(data, bins=bins)
    Lidx, Midx, Ridx = locate_maxima(h, b)
    lsm = locate_stages_middlepoint(h, b, (Lidx, Midx, Ridx), 9, True)

    A, B = lsm['A'], lsm['B']
    return A, B


def run_check(fname):
    # data = read_atf("sample.atf")
    # data = read_atf("data/2019_04_04_00013_20D.atf")
    data = read_atf(fname)
    name = fname[:-4]

    bins = 33 #int(np.sqrt(len(data))) #25
    data = data["C"][data["C"] < 0]
    mean = data.median()

    h, b = histogram(data, bins=bins)

    hprim = np.diff(h)
    bprim = recalculate_bins(b)
    Lidx, Midx, Ridx = locate_maxima(h, b)
    lsm = locate_stages_middlepoint(h, b, (Lidx, Midx, Ridx), 9, True)
    A, B = lsm['A'], lsm['B']

    # historgam plot
    ax = data.hist(bins=bins, normed=True, log='y')
    fig = ax.get_figure()
    # another historgam plot
    plt.plot(b, h, '-o')
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


def detect_drop(data, window=200000, step=20000):
    ld = len(data)
    for idx in range(0, ld - window - step, step):
        a = np.mean(data[idx:idx+window])
        b = np.mean(data[idx+step:idx+window+step])
        print(idx, a, b, is_close(a, b))
        if not is_close(a, b):
            return idx
    return ld


def plot_data(fname, maks=-1):
    data = read_atf(fname)
    name = fname[:-4]
    data = data["C"][data["C"] < 0]
    ax = data[0:maks:10000].plot()
    fig = ax.get_figure()
    fig.savefig(name + '_current.png')
    plt.close()


if __name__ == "__main__":
    fname = "data/2019_04_04_00013_20D.atf"
    data = read_atf(fname)
    data = data["C"][data["C"] < 0]
    dd = detect_drop(data)
    print(dd, len(data))
    plot_data(fname, dd)
    # plot_data("data/2019_04_04_00013_20D.atf")
    # run_check('data/2019_04_04_00015_60D.atf')
    # print(run_sample_check())
    # import glob
    # for fn in glob.glob('data/*D.atf'):
    #     try:
    #         print(fn, plot_data(fn))
    #     except:
    #         print(fn, "...lipa")
