import torch.nn as nn
from DRCNN import DRCNNExp, DRCNNPaper
from helpers import pre_process_and_create_data_loader
from train import train, init_weights_
from predict import predict


def main():
    # Create and initialize the model
    model = DRCNNPaper()
    model = nn.DataParallel(model)
    # init_weights_(model)

    # Create the data loaders
    train_data_loader = pre_process_and_create_data_loader("C:\\Amir\\codes\\Python\\HC_ON_FL\\Data\\Train", batch_size=8)
    valid_data_loader = pre_process_and_create_data_loader("C:\\Amir\\codes\\Python\\HC_ON_FL\\Data\\Valid", batch_size=8)
    test_data_loader = pre_process_and_create_data_loader("C:\\Amir\\codes\\Python\\HC_ON_FL\\Data\\Test", batch_size=8)

    # Perform the training
    best_model_params, _, _, _ = train(model, train_data_loader, valid_data_loader)
    model.load_state_dict(best_model_params)

    # # Load the model parameters
    # best_model_params = torch.load("best_model.pt")
    # model.load_state_dict(best_model_params)

    # Test the model
    predict(model, test_data_loader)


if __name__ == '__main__':
    main()
