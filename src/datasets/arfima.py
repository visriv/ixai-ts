import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

def generate_arfima(seq_len=500, d=0.3):
    # approximate ARFIMA with AR(1) + fractional differencing
    ar = np.array([1, -0.5])
    ma = np.array([1])
    arma = ArmaProcess(ar, ma)
    X = arma.generate_sample(nsample=seq_len)
    # fractional differencing filter
    w = [1]
    for k in range(1, seq_len):
        w.append(w[-1] * ((d - k + 1)) / k)
    w = np.array(w)
    X_fd = np.convolve(X, w, mode="full")[:seq_len]
    return X_fd
