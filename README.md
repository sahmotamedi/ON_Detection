### Prior Optic Neuritis Detection on Peripapillary Ring Scans using Deep Learning

This repository contains the codes used for preparing the data and training and testing the model used for prior optic neuritis detection [paper by Motamedi et al.](https://onlinelibrary.wiley.com/doi/10.1002/acn3.51632)

Additionally, in this repository, codes for grad CAM and some experiments with federated learning can be found which are not part of the paper.

The data used to train, validate, and test the model cannot be published due to GDPR restrictions. 

To train the model, first, the data must be separated into three subfolders: "Train", "Test", and "Valid" each of which has two subfolders "HC" and "ON" that contains OCT .vol files from healthy controls and optic neuritis patients, respectively.

main.py provides an example of how to run the preprocessing, training, and testing of the model. CAM.py shows how grad CAM (class activation map) is created, and FL_HCON_main.py shows a simple federated learning implementation. 

### Requirements
environment.yml
The OCT vol file reader is also needed. You can download it from [here](https://github.com/sahmotamedi/OCT.git).

### Contact
Please contact Amir Motamedi (seyedamirhosein.motamedi(at)charite.de) for any inquiries about this software.

