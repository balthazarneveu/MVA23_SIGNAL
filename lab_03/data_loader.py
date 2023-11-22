
import numpy as np
import h5py  # .h5 data format
from pathlib import Path
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader, Dataset
import torch

DATA_ROOT = Path(__file__).parent/"data"
SAMPLE_DATA_PATH = DATA_ROOT/"samples.hdf5"
TRAIN = "train"
VALID = "validation"
PATH = "path"
BATCH_SIZE = "batch_size"
DEFAULT_BATCH_SIZE = 8
CONFIG_DATAPATHS = {
    TRAIN: {
        PATH: DATA_ROOT/"train.hdf5",
        BATCH_SIZE: DEFAULT_BATCH_SIZE
    },
    VALID: {
        PATH: DATA_ROOT/"validation.hdf5",
        BATCH_SIZE: DEFAULT_BATCH_SIZE
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
    def __init__(self, data_path: Path):
        signals, _snr, labels_id, _label_dict = get_data(data_path)
        self.signals = signals
        self.labels = labels_id

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx: int):
        signal = torch.FloatTensor(self.signals[idx, :])
        label = torch.LongTensor([self.labels[idx]])
        return signal, label


def get_dataloaders(config_data_paths=CONFIG_DATAPATHS
                    ) -> Dict[str, DataLoader]:
    dl_dict = {}
    for mode, config_dict in config_data_paths.items():
        dataset = SignalsDataset(config_dict[PATH])
        dl = DataLoader(dataset, shuffle=False,
                        batch_size=config_dict[BATCH_SIZE])
        dl_dict[mode] = dl
    return dl_dict


if __name__ == '__main__':
    dl_dict = get_dataloaders()
    batch_signal, batch_labels = next(iter(dl_dict[TRAIN]))
