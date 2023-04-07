import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


def init_weights_(model):
    """
    Initialize the model weights

    Parameters
    ----------
    model : torch.nn.Module
        The network for binary classification to be trained.

    Notes
    -----
    This function changes the model by changing its weights
    This function is to experiment different weight initialization (by changing the code), the default initialization
    for Conv2D and Linear is uniform(-/+1/sqrt(n_parameters)), which may be already the best
    """
    # Init weights for all the modules that have learnable weights
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight.data, 1)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)


def train(model, train_data_loader, valid_data_loader, learning_rate=1e-4, n_epochs=50, criterion='BCE',
          optimizer='Adam', stop_early_after=10, decay_lr_every=None,
          print_info=True, plot_losses=True, save_best_model=True, save_name="best_model"):
    """
    Train a model designed for binary classification of healthy controls versus optic neuritis eyes

    Parameters
    ----------
    model : torch.nn.Module
        The network for binary classification to be trained. The output must be the probability, not the score
    train_data_loader : torch.utils.data.DataLoader
        A data loader containing both images and labels for training. Image size must match the input size fo the model
    valid_data_loader : torch.utils.data.DataLoader
        A data loader containing both images and labels for validation. Image size must match the input size fo the model
    learning_rate : float, default 0.0001
        The learning rate of the training (optimizer)
    n_epochs : int, default 50
        The number of epochs for the training
    criterion : str, default 'BCE'
        The loss function. Atm it has only one option which is binary cross entropy
    optimizer : str, default 'Adam'
        The optimizer. Atm it has only one option which is the Adam optimization algorithm
    stop_early_after : int, default 10
        The early stopping 'patience'
    decay_lr_every : int, default None
        If given, the learning rate is divided to 10 every decay_lr_every epoch
    print_info : bool, default True
        If Ture, the model and number of samples in the data loaders are printed
    plot_losses : bool, default True
        If True, the training and validation losses are plotted after the training
    save_best_model : bool, default True
        If true, the state_dict of the best model is saved
    save_name : str, default "best_model"
        The file name for the best model to be saved

    Returns
    -------
    tuple
        The state_disct of the best model (collections.OrderedDict), and three lists containing training loss, validation loss, and validation accuracy of all epochs

    Notes
    -----
    This function doesn't wrap the model in nn.DataParallel class to train the model on multiple GPUs. If needed, the
    model should be wrapped in DataParallel first by model = nn.DataParallel(model) and then passed to this function
    """
    # Print some info about the network and the data if print_info is True
    if print_info:
        print("----------Model---------")
        print(model)
        print("Number of images in the training data set: {}".format(len(train_data_loader.sampler)))
        print("Number of images in the validation data set: {}".format(len(valid_data_loader.sampler)))

    # Define the device that training should be run on
    if torch.cuda.is_available():
        print("Training the model on GPU")
        device = torch.device("cuda")
    else:
        print("Training the model on CPU")
        device = torch.device("cpu")

    # Move the model to the device
    model.to(device)

    # Define optimizer (& scheduler) and loss function
    if criterion == 'BCE':
        criterion = nn.BCELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if decay_lr_every:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_every, gamma=0.1)

    # Define parameters used in the training process
    train_loss = []
    valid_loss = []
    valid_acc = []
    valid_acc_max = -float("inf")
    early_stopping_clock = 0

    for i_epoch in range(n_epochs):
        # Define epoch losses
        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0

        ### Training
        model.train()
        for images, labels in train_data_loader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            # Perform forward and backward pass
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            epoch_train_loss += loss.item()*out.size(0)

        # Calculate the average loss
        train_loss.append(epoch_train_loss/len(train_data_loader.sampler))

        ### Validation
        model.eval()
        correct = 0
        for images, labels in valid_data_loader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            # Perform forward pass and accumulate the loss
            out = model(images)
            loss = criterion(out.squeeze(), labels.squeeze())
            epoch_valid_loss += loss.item()*out.size(0)

            # Calculate the TF+TN for accuracy calculation
            pred = (out.squeeze() >= 0.5).float()
            correct += (pred == labels.squeeze()).float().sum().item()

        # Calculate the average validation loss and accuracy
        valid_loss.append(epoch_valid_loss / len(valid_data_loader.sampler))
        valid_acc.append(correct / len(valid_data_loader.sampler) * 100)

        # Output the results
        print("Epoch {}/{} \t Training loss:{:.6f} \t Validation loss:{:.6f} \t Validation accuracy:{:.3f} %".format(i_epoch+1, n_epochs, train_loss[-1], valid_loss[-1], valid_acc[-1]))

        # Stop early if there are no improvement in validation accuracy for 10 epochs otherwise save the model parameters
        early_stopping_clock += 1
        if valid_acc[-1] >= valid_acc_max:
            print("The validation accuracy increased: {:.3f}% -----> {:.3f}%".format(valid_acc_max, valid_acc[-1]))

            # Update parameters
            valid_acc_max = valid_acc[-1]
            early_stopping_clock = 0

            # Save the parameters of the best model (this code will be run once since any valid_acc is greater than -inf)
            best_model_params = model.state_dict()

        elif early_stopping_clock == stop_early_after:
            break

        # Step scheduler if defined
        if 'scheduler' in locals():
            scheduler.step()
            if (i_epoch + 1) % decay_lr_every == 0:
                print("The learning rate has decayed to {}".format(scheduler.get_last_lr()))

    # Plot losses if plot_losses is True
    if plot_losses:
        plt.plot(train_loss, '-b', label="Training_loss")
        plt.plot(valid_loss, '-r', label="Validation_loss")
        plt.xlabel("Epoch")
        plt.legend(loc='upper right')
        plt.show()

    # Save the state_dict of the best model if save_best_model is True
    if save_best_model:
        torch.save(best_model_params, save_name if '.pt' in save_name else save_name + '.pt')
        print("The best model was saved in '{}'".format(save_name if '.pt' in save_name else save_name + '.pt'))

    # Return losses and accuracies and the best model parameters
    return best_model_params, train_loss, valid_loss, valid_acc
