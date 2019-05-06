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

    hprim = np.diff(hist)
    bprim = recalculate_bins(bins)

    ret = []
    for idx in range(hprim.size - 2):
        if hprim[idx] * hprim[idx+1] < 0 and\
           hprim[idx-1] * hprim[idx+2] < 0 and\
           hprim[idx-1] * hprim[idx] > 0 and\
           hprim[idx+1] * hprim[idx+2] > 0:
                # ret.append(np.mean(bprim[idx-1:idx+3]))
                ret.append(idx + 1)

    if len(ret) > 3:
        warnings.warn("Possible multistable pdf, returning last 3", MultistablePDF)
    return ret[-3:]


def run_sample_check():
    data = read_atf("sample.atf")

    bins = 33#25
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
    #extrema location
    for idx in [Lidx, Midx, Ridx]:
        plt.plot([b[idx]], [np.polyval(lsm['fit'], b[idx])], 'or')
    # polyfit plot
    for x in [A, B]:
        plt.plot([x], [np.polyval(lsm['fit'], x)], 'ok')
    fig.savefig('sample_hist.png')
    plt.close()

    plt.plot(bprim, hprim, 'o-')
    plt.axhline(0, color='k')
    plt.plot([b[Lidx], b[Midx], b[Ridx]], [0] * 3, 'or')
    plt.savefig('sample_histprim.png')


if __name__ == "__main__":
    print(run_sample_check())
