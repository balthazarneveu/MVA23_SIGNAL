
import matplotlib.pyplot as plt


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
    plt.xlabel("real")
    plt.ylabel("imaginary")
    plt.grid()
    plt.show()
