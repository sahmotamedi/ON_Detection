import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from DRCNN import DRCNNPaper
from helpers import pre_process_and_create_data_loader
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from random import randint


class HCONBase:
    def __init__(self):
        # Define the attributes that are common for all subclasses like model and device
        self.model = nn.DataParallel(DRCNNPaper())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8

        # Move the model to GPU and put it in eval mode
        self.model.to(self.device)
        self.model.eval()

    def print_model(self):
        print("----------Model---------")
        print(self.model)

    def print_device(self):
        print("The model is trained on {}".format(self.device.type))

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name if '.pt' in file_name else file_name + '.pt')

    def load_model_from_file(self, file_name):
        self.model.load_state_dict(torch.load(file_name if '.pt' in file_name else file_name + '.pt'))

    def load_model_from_state_dict(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)


class HCONTrainer(HCONBase):
    def __init__(self, train_data_path, valid_data_path):
        super().__init__()

        # Define training pieces like criterion and optimizer
        self.lr = 1e-4
        self.n_epochs = 1
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.decay_lr_every = None
        if self.decay_lr_every:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.decay_lr_every, gamma=0.1)
        self.enable_prints = True

        # Create data loaders
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.train_data_loader = pre_process_and_create_data_loader(self.train_data_path, batch_size=self.batch_size)
        self.valid_data_loader = pre_process_and_create_data_loader(self.valid_data_path, batch_size=self.batch_size)

        # Define parameters used in the training process
        self.current_lr = self.lr
        self.previous_lr = self.lr
        self.train_loss = []
        self.valid_loss = []
        self.valid_acc = []
        self.valid_acc_max = -float("inf")
        self.valid_acc_no_improvement_count = 0
        self.training_count = 0     # This is a counter to keep track of how many times the model training was executed
        self.validation_count = 0
        self.best_model_state_dict = self.model.state_dict()
        self.trainer_id = randint(1000000000, 9999999999)

    def initialize_weights(self):
        # Init weights for all the modules that have learnable weights
        for module in self.model.modules():
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

    def train(self):
        for i_epoch in range(self.n_epochs):
            # Train
            self.model.train()
            self.training_count += 1
            epoch_train_loss = 0.0
            for images, labels in self.train_data_loader:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Perform forward and backward pass
                self.optimizer.zero_grad()
                out = self.model(images)
                loss = self.criterion(out.squeeze(), labels.squeeze())
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss
                epoch_train_loss += loss.item() * out.size(0)

            # Calculate the average loss
            self.train_loss.append(epoch_train_loss / len(self.train_data_loader.sampler))

            # Print train loss
            if self.enable_prints:
                print("Trainer {} \t Training round {} \t Training loss:{:.6f}".format(self.trainer_id, self.training_count, self.train_loss[-1]))

            # Step scheduler if enabled
            if self.decay_lr_every:
                self.scheduler.step()
                self.previous_lr = self.current_lr
                self.current_lr = self.scheduler.get_last_lr()[0]
                if self.current_lr != self.previous_lr and self.enable_prints:
                    print("Trainer {} \t the learning rate has decayed from {} to {}".format(self.trainer_id, self.previous_lr, self.current_lr))

    def validate(self):
        # Validate
        self.validation_count += 1
        self.model.eval()
        epoch_valid_loss = 0.0
        correct = 0
        for images, labels in self.valid_data_loader:
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Perform forward pass and accumulate the loss
            out = self.model(images)
            loss = self.criterion(out.squeeze(), labels.squeeze())
            epoch_valid_loss += loss.item() * out.size(0)

            # Calculate the TF+TN for accuracy calculation
            pred = (out.squeeze() >= 0.5).float()
            correct += (pred == labels.squeeze()).float().sum().item()

        # Calculate the average validation loss and accuracy
        self.valid_loss.append(epoch_valid_loss / len(self.valid_data_loader.sampler))
        self.valid_acc.append(correct / len(self.valid_data_loader.sampler) * 100)

        # Print validation info
        if self.enable_prints:
            print("Trainer {} \t Validation round {} \t Validation loss:{:.6f} \t Validation accuracy:{:.3f}"
                  .format(self.trainer_id, self.validation_count, self.valid_loss[-1], self.valid_acc[-1]))

        # Save the best model if there is an improvement in validation accuracy
        self.valid_acc_no_improvement_count += 1
        if self.valid_acc[-1] >= self.valid_acc_max:
            if self.enable_prints:
                print("Trainer {} \t The validation accuracy increased: {:.3f}% -----> {:.3f}%".format(self.trainer_id, self.valid_acc_max, self.valid_acc[-1]))

            # Update parameters
            self.valid_acc_max = self.valid_acc[-1]
            self.valid_acc_no_improvement_count = 0

            # Save the parameters of the best model
            self.best_model_state_dict = self.model.state_dict()

    def plot_losses(self):
        plt.plot(self.train_loss, '-b', label="Training_loss")
        plt.plot(self.valid_loss, '-r', label="Validation_loss")
        plt.title("Trainer {}".format(self.trainer_id))
        plt.xlabel("Epoch")
        plt.legend(loc='upper right')
        plt.show()

    def load_best_model(self):
        self.load_model_from_state_dict(self.best_model_state_dict)


class HCONPredictor(HCONBase):
    def __init__(self, test_data_path):
        super().__init__()

        # Create the data loaders
        self.test_data_path = test_data_path
        self.test_data_loader = pre_process_and_create_data_loader(self.test_data_path, batch_size=self.batch_size)

        # Define results lists
        self.ground_truth = []
        self.prediction = []

        # Define metrics
        self.tp = self.tn = self.fn = self.fp = self.acc = self.sens = self.spec = self.prec = self.f1 = None

    def predict(self):
        # Perform the prediction
        self.model.eval()
        for images, labels in self.test_data_loader:
            images = images.to(self.device)
            self.ground_truth.extend(labels.squeeze().tolist())
            out = self.model(images)
            self.prediction.extend((out.squeeze() >= 0.5).float().tolist())

        # Calculate metrics
        self._calculate_metrics()

        # Return the instance to allow method cascading
        return self

    def _calculate_metrics(self):
        # Calculate different metrics
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(np.array(self.ground_truth),
                                                              np.array(self.prediction)).ravel()
        self.acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        self.sens = self.tp / (self.tp + self.fn)
        self.spec = self.tn / (self.tn + self.fp)
        self.prec = self.tp / (self.tp + self.fp)
        self.f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)

    def print_metrics(self):
        print("Positive = {}, Negative = {}, Total = {}".format(
            self.tp + self.fn, self.tn + self.fp, self.tp + self.tn + self.fp + self.fn))
        print(
            "Prediction confusion matrix: True Positive = {}, False Negative = {}, False Positive = {}, True Negative = {}".format(
                self.tp, self.fn, self.fp, self.tn))
        print(
            "Prediction metrics: Accuracy = {:.4f}, Sensitivity = {:.4f}, Specificity = {:.4f}, Precision = {:.4f}, F1_score = {:.4f}".format(
                self.acc, self.sens, self.spec, self.prec, self.f1))
