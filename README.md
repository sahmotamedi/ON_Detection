### Prior Optic Neuritis Detection on Peripapillary Ring Scans using Deep Learning

This repository contains the code used to prepare the data and train and test the model for the paper on prior optic neuritis detection by [Motamedi et al.](https://onlinelibrary.wiley.com/doi/10.1002/acn3.51632)

In addition, this repository includes code for grad CAM and some experiments with federated learning, which are not part of the paper.

The data used to train, validate, and test the model cannot be published due to GDPR restrictions.

To train the model, the data must first be separated into three subfolders: "Train", "Test", and "Valid", each of which has two subfolders "HC" and "ON" containing OCT .vol files from healthy controls and optic neuritis patients, respectively.

The 'main.py' file provides an example of how to run the preprocessing, training, and testing of the model. The 'CAM.py' file shows how grad CAM (class activation map) is created, and the 'FL_HCON_main.py' file shows a simple federated learning implementation.

### Requirements
The environment.yml file lists the required dependencies. The OCT vol file reader is also needed, and it can be downloaded from [here](https://github.com/sahmotamedi/OCT.git).

### Contact
If you have any questions or comments, please contact us at seyedamirhosein.motamedi(at)charite.de.

