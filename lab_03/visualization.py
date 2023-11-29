
import matplotlib.pyplot as plt
import numpy as np


def visualize_signals(data_in: list = None,
                      idx: int = 0):
    """Visualize signals, better used interactively with ipywidget

    Args:
        data_in (list): data structure as a list 
        [signals, snr, labels_id, label_dict]
        idx (int, optional): Specific index of the signal to plot.
        Defaults to 0.

    Usage:
    ======
    Use with ipywidget interactive like in a notebook
    ```
    from ipywidgets import interact, IntSlider, fixed
    interact(
        visualize_signals,
        data_in = fixed([signals, snr, labels_id, label_dict]),
        idx=IntSlider(min=0, max=signals.shape[0]-1, step=1)
    )
    ```
    """
    signals, snr, labels_id, label_dict = data_in
    plt.figure(figsize=(8, 8))
    plt.scatter(signals[idx, :, 0], signals[idx, :, 1],
                label="complex", alpha=0.5)
    plt.title(f'Index {idx} - {label_dict.get(labels_id[idx], "unknown")}'
              + f' SNR={snr[idx]} db')
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.show()

def visualize_two_signals(data_1: list = None,
                      data_2: list = None,
                      idx: int = 0):
    """Visualize signals, better used interactively with ipywidget

    Args:
        data_1 (list): data structure as a list 
        [signals, snr, labels_id, label_dict]
        data_2 (list): data structure as a list 
        [signals, snr, labels_id, label_dict]
        idx (int, optional): Specific index of the signal to plot.
        Defaults to 0.

    Usage:
    ======
    Use with ipywidget interactive like in a notebook
    ```
    from ipywidgets import interact, IntSlider, fixed
    interact(
        visualize_signals,
        data_1 = fixed([signals, snr, labels_id, label_dict]),
        data_2 = fixed([signals_2, snr_2, labels_id_2, label_dict_2]),
        idx=IntSlider(min=0, max=signals.shape[0]-1, step=1)
    )
    ```
    """
    signals, snr, labels_id, label_dict = data_1
    signals_2, snr_2, labels_id_2, label_dict_2 = data_2
    fig, axs = plt.subplots(2, sharex = True, sharey = True)
    axs[0].scatter(signals[idx, :, 0], signals[idx, :, 1],
                label="complex", alpha=0.5)
    axs[1].scatter(signals_2[idx, :, 0], signals_2[idx, :, 1],
                label="complex", alpha=0.5)
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    plt.suptitle(f'Index {idx} - {label_dict.get(labels_id[idx], "unknown")}'
              + f' SNR={snr[idx]} db')
    plt.show()
