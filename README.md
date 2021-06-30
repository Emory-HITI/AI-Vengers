# AI-Vengers

This GitHub repository holds training and validation code for Deep Learning and Machine Learning models to detect racial demographics of patients through the use of medical images.

The **data** and **models** folders will be empty as -
1. The data and model file occupy a large amount of data and cannot be pushed onto GitHub repositories.
2. Some of the data used to conduct experiments is proprietary. URLs were attached for the open-source datasets. 
3. The trained ML/DL models can be used to re-create the patient information unto certain levels which could leak the proprietary data.

The training data folder has training code for all the experiments. The experiments, corresponding data and model were as following -

| Training Folder Name | Training File Name | Data | Model |
| -------------------- | ------------------ | ---- | ----- |
| CXR_training         | CheXpert_resnet34_race_detection_2021_06_29.ipynb  | CheXpert           | ResNet34    |
| CXR_training         | Emory_CXR_resnet34_race_detection_2021_06_29.ipynb | Emory CXR          | ResNet34    |
| CXR_training         | MIMIC_resnet34_race_detection_2021_06_29.ipynb     | MIMIC              | Resnet34    |
| EM-CS_training       | Emory_C-spine_race_detection_2021_06_29.ipynb      | Emory Cervical Spine | Resnet34  |
| EM_Mammo_Training    | training code.ipynb                                | Mammogram          | EfficientNetB2 |
| Densenet121_CXR_Training | Lung_segmentation_MIMIC.ipynb                  | MIMIC              |             |
| Densenet121_CXR_Training | Race classification with No Finding label only_MIMIC_Densenet121.ipynb | MIMIC | DenseNet121 |
| Densenet121_CXR_Training | Race classification_MIMIC_Densenet121.ipynb    | MIMIC              | DenseNet121 |
| Densenet121_CXR_Training | Race_classification_Emory_Densenet121.ipynb    | Emory CXR          | DenseNet121 |
| digital_hand_atlas   | dha_2_classes.ipynb                                | Digital Hand Atlas | ResNet50    |
| frequency_training   |                                                    |                    |             |

The final ipython-notebook — bias_pred.ipynb has validation code for all the above training models (except frequency training).

To run the validation code -
1. Fork/Download the GitHub repository.
2. Fetch the data from the data URLs for open-source datasets and drop them in the data folder.
3. Run the corresponding training code and save the trained model in the models folder. 
4. Change the model path in the validation code and the corresponding function.

https://emory-hiti.github.io/AI-Vengers/
