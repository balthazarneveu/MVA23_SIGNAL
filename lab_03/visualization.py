
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def visualize_signals(data_in=None,
                      idx: int = 0):
    signals, snr, labels_id, label_dict = data_in
    plt.figure(figsize=(8, 8))
    # plt.plot(signals[idx, :, 0], label="real")
    # plt.plot(signals[idx, :, 1], label="imaginary")
    plt.scatter(signals[idx, :, 0], signals[idx, :, 1],
                label="complex", alpha=0.5)
    plt.title(f'Index {idx} - {label_dict.get(labels_id[idx], "unknown")}'
              + f'SNR={snr[idx]} db')
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.show()
