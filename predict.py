import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def predict(model, data_loader, print_info=True, print_metrics=True):
    """
    Predict the class of an input OCT image (HC=0 or ON=1)

    Parameters
    ----------
    model : nn.Module
        The model
    data_loader : torch.utils.data.DataLoader
        A data loader containing both images and labels. Image size must match the input size fo the model
    print_info : bool, default True
        If Ture, the model and number of samples in the data loader are printed
    print_metrics : bool, default True
        If True the classification metrics (accuracy, sensitivity, etc) are printed

    Returns
    -------
    tuple
        Two np.ndarray containing the ground truth (labels) and class prediction for all the samples in the data loader

    Notes
    -----
    This function doesn't wrap the model in nn.DataParallel class to train the model on multiple GPUs. If needed, the
    model should be wrapped in DataParallel first by model = nn.DataParallel(model) and then passed to this function
    """
    # Print some info about the network and the data if print_info is True
    if print_info:
        print("----------Model---------")
        print(model)
        print("Number of images in the prediction data set: {}".format(len(data_loader.sampler)))

    # Define the device that training should be run on
    if torch.cuda.is_available():
        print("Running the model on GPU")
        device = torch.device("cuda")
    else:
        print("Running the model on CPU")
        device = torch.device("cpu")

    # Move the model to the device
    model.to(device)

    # Define results lists
    ground_truth = []
    prediction = []

    # Perform the prediction
    model.eval()
    for images, labels in data_loader:
        # Move to device
        images = images.to(device)

        # Store the ground truth
        ground_truth.extend(labels.squeeze().tolist())

        # Perform forward pass
        out = model(images)

        # Convert output to prediction class
        prediction.extend((out.squeeze() >= 0.5).float().tolist())

    # Transfer to numpy
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Print classification metrics if print_metrics is True
    if print_metrics:
        tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
        print("Positive = {}, Negative = {}, Total = {}".format(tp+fn, tn+fp, tp+tn+fp+fn))
        print("Prediction confusion matrix: True Positive = {}, False Negative = {}, False Positive = {}, True Negative = {}".format(tp, fn, fp, tn))
        print("Prediction metrics: Accuracy = {:.4f}, Sensitivity = {:.4f}, Specificity = {:.4f}, F1_score = {:.4f}".format((tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tn/(tn+fp), 2*tp/(2*tp+fp+fn)))

    # Return the prediction and ground truth
    return ground_truth, prediction
