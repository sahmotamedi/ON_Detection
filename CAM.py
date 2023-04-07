from DRCNN_CAM import DRCNNPaper
import torch
import torch.nn as nn
from helpers import pre_process_and_create_data_loader
import matplotlib.pyplot as plt
import skimage.transform as skt


def generate_cam():
    # Define the model and load it with pre-trained parameters
    model = nn.DataParallel(DRCNNPaper())
    model.load_state_dict(torch.load('best_model.pt'))
    device = torch.device('cuda')

    # Take out only one image from test data
    test_data_loader = pre_process_and_create_data_loader("C:\\Amir\\codes\\Python\\HC_ON_FL\\Data\\Test", batch_size=1)

    img, labels = next(iter(test_data_loader))
    img = img.to(device)

    # Perform forward and backward in eval mode
    model.eval()
    pred = model(img).squeeze()
    pred.backward()

    # Get the gradient and the last activation and scale them according to the gradients
    gradients = model.module.get_activation_gradients()
    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.module.get_activations(img).detach()
    for i_channel in range(pooled_grad.size(0)):
        activations[:, i_channel, :, :] *= pooled_grad[i_channel]

    # Accumulate all channels, normalize, and reshape
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = heatmap.relu()
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    heatmap = skt.resize(heatmap, (512, 512))

    # Print the label, the prediction and show the heatmap together with the image
    print(labels.squeeze())
    print(pred.detach().cpu())
    img = img.cpu().numpy()
    plt.imshow(img[0, 0, :, :], cmap='gray')
    plt.imshow(heatmap, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    generate_cam()
