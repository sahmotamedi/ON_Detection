from HCON import HCONTrainer, HCONPredictor
from matplotlib import pyplot as plt


def fl_train():
    n_rounds = 30

    # Define the variables needed to save the best model
    total_valid_loss = []
    total_valid_acc = []
    total_valid_acc_max = -float('inf')
    best_model_state_dict = dict()

    # Create the models
    trainer_1 = HCONTrainer(r"C:\Amir\codes\Python\HC_ON_FL\Data2\Site1\Train",
                            r"C:\Amir\codes\Python\HC_ON_FL\Data2\Site1\Valid")
    trainer_2 = HCONTrainer(r"C:\Amir\codes\Python\HC_ON_FL\Data2\Site2\Train",
                            r"C:\Amir\codes\Python\HC_ON_FL\Data2\Site2\Valid")

    # Print the trainer ids
    print("Trainer 1 id is {}".format(trainer_1.trainer_id))
    print("Trainer 2 id is {}".format(trainer_2.trainer_id))

    # Initialize the trainer_1 and copy this initialization to other trainers
    # trainer_1.initialize_weights()      # The default weight initialization seem to converge faster and better
    trainer_2.load_model_from_state_dict(trainer_1.model.state_dict())

    # Assign the weights according to each trainer training data share
    trainer_1_train_data_size = len(trainer_1.train_data_loader.sampler)
    trainer_2_train_data_size = len(trainer_2.train_data_loader.sampler)
    total_training_data = trainer_1_train_data_size + trainer_2_train_data_size
    train_weight_1 = trainer_1_train_data_size / total_training_data
    train_weight_2 = trainer_2_train_data_size / total_training_data

    # Calculate the proportion of the validation data sets
    trainer_1_valid_data_size = len(trainer_1.valid_data_loader.sampler)
    trainer_2_valid_data_size = len(trainer_2.valid_data_loader.sampler)
    total_validation_data = trainer_1_valid_data_size + trainer_2_valid_data_size
    valid_weight_1 = trainer_1_valid_data_size / total_validation_data
    valid_weight_2 = trainer_2_valid_data_size / total_validation_data

    for i_round in range(n_rounds):
        # Train for 1 epoch
        trainer_1.train()
        trainer_2.train()

        # Aggregate by doing federated averaging
        state_dict_model_1 = trainer_1.model.state_dict()
        state_dict_model_2 = trainer_2.model.state_dict()
        for name in state_dict_model_1.keys():
            state_dict_model_1[name].data.copy_(train_weight_1*state_dict_model_1[name].data + train_weight_2*state_dict_model_2[name].data)

        # Update the weights
        trainer_1.load_model_from_state_dict(state_dict_model_1)
        trainer_2.load_model_from_state_dict(state_dict_model_1)

        # Validate
        trainer_1.validate()
        trainer_2.validate()

        # Add the validation loss and accuracy for this round and print them
        total_valid_acc.append(valid_weight_1*trainer_1.valid_acc[-1] + valid_weight_2*trainer_2.valid_acc[-1])
        total_valid_loss.append(valid_weight_1*trainer_1.valid_loss[-1] + valid_weight_2*trainer_2.valid_loss[-1])
        print("Round {}/{} \t Total validation loss:{:.6f} \t Total validation accuracy:{:.3f}".format(i_round+1, n_rounds, total_valid_loss[-1], total_valid_acc[-1]))

        # Save the model if there is an improvement in total validation accuracy
        if total_valid_acc[-1] >= total_valid_acc_max:
            print("For FL, The validation accuracy increased: {:.3f}% -----> {:.3f}%".format(total_valid_acc_max, total_valid_acc[-1]))

            # Update parameters
            total_valid_acc_max = total_valid_acc[-1]

            # Save the parameters of the best model
            best_model_state_dict = trainer_1.model.state_dict()

    # Plot the accuracies
    plt.plot(total_valid_acc)
    plt.show()

    # Save the model
    trainer_1.load_model_from_state_dict(best_model_state_dict)
    trainer_1.save_model('Best_FL_model')

    # Predict
    predictor = HCONPredictor(r"C:\Amir\codes\Python\HC_ON_FL\Data2\Test")
    predictor.load_model_from_file("Best_FL_model")
    predictor.predict()
    predictor.print_metrics()


def single_train():
    n_rounds = 30

    # Create the models
    trainer = HCONTrainer(r"C:\Amir\codes\Python\HC_ON_FL\Data\Train",
                          r"C:\Amir\codes\Python\HC_ON_FL\Data\Valid")

    # trainer.initialize_weights()      # The default weight initialization seem to converge faster and better

    for i_round in range(n_rounds):
        # Train for 1 epoch
        trainer.train()

        # Validate
        trainer.validate()

    # Plot the accuracies
    trainer.plot_losses()
    plt.plot(trainer.valid_acc)
    plt.show()

    # Save the model
    trainer.load_model_from_state_dict(trainer.best_model_state_dict)
    trainer.save_model('Best_Single_model')

    # Predict
    predictor = HCONPredictor(r"C:\Amir\codes\Python\HC_ON_FL\Data\Test")
    predictor.load_model_from_file("Best_Single_model")
    predictor.predict()
    predictor.print_metrics()


if __name__ == '__main__':
    fl_train()
