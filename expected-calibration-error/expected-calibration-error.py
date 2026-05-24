import numpy as np
from collections import defaultdict
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    bins = defaultdict(list)
    bins_p = defaultdict(list)
    for i, p in enumerate(y_pred):
        indx = int(p * n_bins)
        bins[indx].append(y_true[i])
        bins_p[indx].append(y_pred[i])
    accs = [(sum(ls) / len(ls), len(ls)) for k, ls in bins.items()]
    confs = [(sum(ls) / len(ls), len(ls)) for k, ls in bins_p.items()]

    fn = [(v[1] / len(y_true)) * abs(v[0] - conf[0]) for v, conf in zip(accs, confs)]
    return sum(fn)