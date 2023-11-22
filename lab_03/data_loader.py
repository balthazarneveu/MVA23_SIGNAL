
import numpy as np
import h5py  # .h5 data format
from pathlib import Path
from typing import Tuple, Optional, Dict
DATA_ROOT = Path(__file__).parent/"data"


def get_data(data_root: Optional[Path] = DATA_ROOT) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """Get signals and metadata (snr, labels) + correspondance dictionary
    from a h5 file containing all data.

    Args:
        data_root (Optional[Path], optional): path to .h5 file.
        Defaults to DATA_ROOT so you don't have to worry about providing it

    Returns:
        Tuple[ np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        signals, snr, labels_id, label_dict

    """
    data_path = data_root/"samples.hdf5"
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

