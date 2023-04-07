import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import pre_process_batch


class DirNotFoundError(Exception): pass


def pre_process_and_create_data_loader(data_path, batch_size, num_workers=0):
    """
    Pre-process the data in both HC and ON cohorts, combine them and return the data loader

    Parameters
    ----------
    data_path : str
        The path to a folder containing two sub folders "HC" and "ON"
    batch_size : int
        The batch size for the data loader
    num_workers : int, default 0
        The number of workers for the data loader

    Raises
    ------
    DirNotFoundError
        If the data_path doesn't have sub folders named "HC" and "ON"

    Returns
    -------
    torch.utils.data.DataLoader
        A data loader that returns tuple(images, labels) at each iteration with the axis 0 size of the batch size
    """
    # Check if the sub folders with the name "HC" and "ON" exist, if not, raise an error
    if not os.path.isdir(os.path.join(data_path, "HC")) or not os.path.isdir(os.path.join(data_path, "ON")):
        raise DirNotFoundError("The data path must point to a folder with two sub folders named 'HC' and 'ON'!")

    # Read and preprocess the HC and ON data
    HC_data = pre_process_batch(os.path.join(data_path, "HC"))
    ON_data = pre_process_batch(os.path.join(data_path, "ON"))

    # Transform the data to torch Tensor
    HC_tensor = torch.from_numpy(HC_data)
    ON_tensor = torch.from_numpy(ON_data)
    del HC_data, ON_data

    # Create labels
    HC_label = torch.zeros((HC_tensor.size()[0], 1))
    ON_label = torch.ones((ON_tensor.size()[0], 1))

    # Create the tensor dataset
    data_set = TensorDataset(torch.cat((HC_tensor, ON_tensor), dim=0), torch.cat((HC_label, ON_label), dim=0))
    del HC_tensor, ON_tensor

    # Create the data loader and return
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

