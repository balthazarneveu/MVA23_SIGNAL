
import numpy as np
import h5py  # .h5 data format
from pathlib import Path
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader, Dataset
import torch
import logging
import matplotlib.pyplot as plt
DATA_ROOT = Path(__file__).parent/"data"
SAMPLE_DATA_PATH = DATA_ROOT/"samples.hdf5"
TRAIN = "train"
VALID = "validation"
PATH = "path"
BATCH_SIZE = "batch_size"
SHUFFLE = "shuffle"
AUGMENT_TRIM = "augment_trim"
AUGMENT_NOISE = "augment_noise"
AUGMENT_ROTATE = "augment_rotate"
SNR_FILTER = "snr_filter"

DEFAULT_BATCH_SIZE = 8
CONFIG_DATALOADER = {
    TRAIN: {
        PATH: DATA_ROOT/"train.hdf5",
        BATCH_SIZE: DEFAULT_BATCH_SIZE,
        SHUFFLE: True,
        AUGMENT_TRIM: False,  # These can be modified
        AUGMENT_NOISE: 0,
        AUGMENT_ROTATE: False,
        SNR_FILTER: None,
    },
    VALID: {
        PATH: DATA_ROOT/"validation.hdf5",
        BATCH_SIZE: DEFAULT_BATCH_SIZE,
        SHUFFLE: False,
        AUGMENT_TRIM: False,
        AUGMENT_NOISE: 0,
        AUGMENT_ROTATE: False,
        SNR_FILTER: None
    }
}


def get_data(data_path: Optional[Path] = SAMPLE_DATA_PATH) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Get signals and metadata (snr, labels) + correspondance dictionary
    from a h5 file containing all data.

    Args:
        data_path (Optional[Path], optional): path to .h5 file.
        Defaults to SAMPLE_DATA_PATH
        so you don't have to worry about providing it

    Returns:
        Tuple[ np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        signals, snr, labels_id, label_dict

    """
    assert data_path.exists()
    with h5py.File(data_path, 'r') as data:
        signals = np.array(data['signaux'])
        snr = np.array(data['snr'])
        labels_id = np.array(data['labels'])
        label_dict = get_labels(data)
    logging.info(f"Loaded {len(signals)} signals from {data_path}")
    return signals, snr, labels_id, label_dict


def get_labels(data_dict: dict) -> Dict[int, str]:
    """Get the mapping between a label_id and the nice label name
    {
        1: 'N-PSK8',
        0: 'N-QAM16',
        2: 'N-QPSK',
        4: 'W-PSK8-V1',
        5: 'W-PSK8-V2',
        3: 'W-QAM16'
    }
    @TODO: describe the meaning of each element
    N/W 
    PSK/QAM
    8/16

    Args:
        data_dict (dict): dictionary loaded from the h5 file.

    Returns:
        dict Dict[int, str]: {label_id: label name}

    """
    return {
        data_dict['label_name'].attrs[k]: k
        for k in data_dict['label_name'].attrs.keys()
    }


class SignalsDataset(Dataset):
    """Extract training/valid data from a .hdf5 file
    Prepares batches of tensors of size [N, C, L]
    N = batch size
    C = number of channels, here C=2 for real and imaginary parts
    (similar to C=3 colors in colored RGB images for instance)
    L = length of the signal
    (similar to [H,W] for images)
    """

    def __init__(
        self,
        data_path: Path,
        augment_trim: Optional[bool] = False,
        augment_noise: Optional[bool] = 0,
        augment_rotate: Optional[bool] = False,
        debug_augmentation_plot: Optional[bool] = False,
        snr_filter: Optional[float] = None
    ):
        if augment_trim:
            logging.warning("ENABLED AUGMENTATION: Trim")
        self.augment_trim = augment_trim
        if augment_noise:
            logging.warning("ENABLED AUGMENTATION: Add noise")
        self.augment_noise = augment_noise
        if augment_rotate:
            logging.warning("ENABLED AUGMENTATION: Rotate")
        self.augment_rotate = augment_rotate
        signals, snr, labels_id, _label_dict = get_data(data_path)
        if snr_filter is not None:
            signals_f, labels_id_f, snr_f = [], [], []
            for snr_filt in snr_filter:
                signals_f.extend(signals[snr == snr_filt])
                labels_id_f.extend(labels_id[snr == snr_filt])
                snr_f.extend(snr[snr == snr_filt])
            signals = np.array(signals_f)
            labels_id = np.array(labels_id_f)
            snr = np.array(snr_f)
        self.signals = signals
        self.labels = labels_id
        self.debug_augmentation_plot = debug_augmentation_plot

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        signal = torch.FloatTensor(self.signals[idx, :].T)  # [2, L]

        # AUGMENTATIONS DEFINITIONS
        if self.augment_trim:
            # AUGMENT: Trim
            # Trim start of sequence between 0 and  100
            start = torch.randint(0, 100, (1,))
            length = int((signal.shape[1]//2) +
                         abs((signal.shape[1])/4 + signal.shape[1]/4 * torch.randn((1,)))
                         )
            signal = signal[:, start:min(start+length, signal.shape[1])]
        if self.augment_rotate:
            # AUGMENT: Rotation
            angle_deg = torch.rand(1) * 360.
            phi = torch.deg2rad(angle_deg)
            s = torch.sin(phi)
            c = torch.cos(phi)
            rot = torch.Tensor([[c, -s], [s, c]])
            if self.debug_augmentation_plot:
                trim_plot = 200
                plt.plot(signal[0, :trim_plot],
                         signal[1, :trim_plot], ".", label="original")
                signal_rot = torch.mm(rot, signal)
                plt.plot(signal_rot[0, :trim_plot],
                         signal_rot[1, :trim_plot], "+", label="augmented")
                for el in range(trim_plot):
                    plt.plot([signal[0, el], signal_rot[0, el]],
                             [signal[1, el], signal_rot[1, el]], "r-", alpha=0.5)
                plt.legend()
                plt.axis("equal")
                plt.grid()
                plt.title(
                    f'Signal augmentation by rotation {float(angle_deg):.1f}° - first {trim_plot} samples')
                plt.show()

            signal = torch.mm(rot, signal)
        if self.augment_noise:
            # AUGMENT: Add noise
            # AWGN (additive white gaussian noise)
            # with a standard deviation sampled
            # uniformly from [0, augment_noise]
            # Each signal has a different noise level
            std_dev = torch.rand(1)*self.augment_noise
            std_dev = std_dev.repeat(2, 1)
            signal += torch.randn(signal.shape)*std_dev
        label = torch.LongTensor([self.labels[idx]])
        return signal, label


def signals_collate_pad_2048_fn(batch):
    """Collate function for SignalsDataset that pads signals 
    to a fixed length of 2048.

    Args:
    batch (list): A list of tuples, where each tuple contains the signal and label.

    Returns:
    Tensor: A batch of signals, padded to the length of 2048.
    Tensor: A batch of labels.
    """
    signals, labels = zip(*batch)

    # Pad each signal to the length of 2048
    padded_signals = [torch.cat([signal, torch.zeros(2, 2048 - signal.shape[1])], dim=1)
                      if signal.shape[1] < 2048 else signal for signal in signals]

    # Stack the signals into a single tensor
    padded_signals = torch.stack(padded_signals)

    # Stack the labels into a single tensor
    labels = torch.stack(labels)

    return padded_signals, labels


def signals_collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for SignalsDataset = 
    trims signals to the minimum length in the batch.

    Args:
    batch (list): A list of tuples, where each tuple contains the signal and label.

    Returns:
    - Tensor: A batch of signals, trimmed to the same (minimum) length.
    - Tensor: A batch of labels.
    """

    signals, labels = zip(*batch)

    # Find the length of the shortest signal in the batch
    min_length = min(signal.shape[1] for signal in signals)
    min_length = min_length-(min_length % 16) # Get a multiple of 16

    # Trim all signals to the minimum length
    trimmed_signals = torch.stack(
        [signal[:, :min_length] for signal in signals], axis=0)
    # Stack the labels into a single tensor
    labels = torch.stack(labels)

    return trimmed_signals, labels


def get_dataloaders(config_data_paths: dict = CONFIG_DATALOADER,
                    ) -> Dict[str, DataLoader]:
    """Instantiates train and valid dataloaders

    Args:
        config_data_paths (dict, optional): Configuration for train/valid
        containing paths & batch size.
        Defaults to CONFIG_DATALOADER.

    Returns:
        Dict[str, DataLoader]: A dictionary withs dataloaders
        for training and validation.
    """
    dl_dict = {}
    for mode, config_dict in config_data_paths.items():
        logging.info(config_dict)
        dataset = SignalsDataset(
            config_dict[PATH],
            augment_trim=config_dict.get(AUGMENT_TRIM, False),
            augment_noise=config_dict.get(AUGMENT_NOISE, 0),
            augment_rotate=config_dict.get(AUGMENT_ROTATE, False),
            snr_filter=config_dict.get(SNR_FILTER, None)
        )
        dl = DataLoader(
            dataset,
            shuffle=config_dict[SHUFFLE],
            batch_size=config_dict[BATCH_SIZE],
            collate_fn=signals_collate_fn if config_dict.get(
                AUGMENT_TRIM, False) else None
        )
        dl_dict[mode] = dl
    return dl_dict


if __name__ == '__main__':
    from copy import deepcopy
    config_data_paths = deepcopy(CONFIG_DATALOADER)
    config_data_paths[TRAIN][SNR_FILTER] = [
        0,]  # [10, 20, 30]  # Select some SNR
    dl_dict = get_dataloaders(config_data_paths)
    print(len(dl_dict[TRAIN].dataset))
    batch_signal, batch_labels = next(iter(dl_dict[TRAIN]))
