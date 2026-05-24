import numpy as np 

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    # Write code here
    # np.linalg.inv(X.T @ X) @ X.T @ y
    cal_labels = np.array(cal_labels).astype(np.float64)
    cal_probs = np.array(cal_probs)
    new_probs = np.array(new_probs)
    for i, j in zip(range(len(cal_labels)), range(1, len(cal_labels))):
        y_bef = cal_labels[i]
        y_next = cal_labels[j]
        if y_bef > y_next:
            avg = (y_bef + y_next) / 2
            cal_labels[i] = cal_labels[j] = avg
    res = []
    for i in range(len(cal_labels)):
        for pred in new_probs:
            if cal_probs[i] > pred and (not i or cal_probs[i - 1] <= pred):
                prev_lab = 0 if not i else cal_labels[i-1]
                prev_prob = 0 if not i else cal_probs[i-1]
                new_val = prev_lab + ((pred - prev_prob) / (cal_probs[i] - prev_prob)) * (cal_labels[i] - prev_lab)
                res.append(new_val)
            if i == (len(cal_labels) - 1) and cal_probs[i] <= pred:
                res.append(cal_labels[i])
    return res
                