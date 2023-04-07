import pytest
from HCON import HCONBase, HCONTrainer
import torch


@pytest.fixture
def trainer():
    return HCONTrainer(r"C:\Amir\codes\Python\HC_ON_FL\Data\Train", r"C:\Amir\codes\Python\HC_ON_FL\Data\Valid")


@pytest.fixture
def tensor_rand(trainer):
    return torch.randn((20, 2, 512, 512), requires_grad=False).to(trainer.device)


def test_init(trainer):
    isinstance(trainer, HCONBase)
    assert trainer.batch_size == 8


# Test the output of the model
def test_output(trainer, tensor_rand):
    assert trainer.model(tensor_rand).squeeze().size(0) == 20
    assert trainer.model(tensor_rand).squeeze().ndim == 1
    assert all((trainer.model(tensor_rand).squeeze() <= 1).tolist()) and all((trainer.model(tensor_rand).squeeze() >= 0).tolist())
