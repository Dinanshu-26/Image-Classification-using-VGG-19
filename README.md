# Image Classification using VGG-19

## Project Overview

This project implements an image classification system using the **VGG19** model, a pre-trained Convolutional Neural Network (CNN) known for its excellent performance in image classification tasks. The model is fine-tuned and trained on a custom dataset to classify images into various categories. The dataset is preprocessed, augmented, and then fed into the model for training.

The project uses **TensorFlow** and **Keras** libraries, making it easy to modify and train the model further. The model is evaluated using various performance metrics such as accuracy, F1 score, precision, recall, and confusion matrix to assess its performance on the test set.

## Table of Contents

- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Project Setup](#project-setup)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Results](#results)

## Dependencies

This project requires the following Python libraries:

- **TensorFlow**
- **Keras**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **zipfile** (Python built-in)

To install these dependencies, you can create a virtual environment and install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
## Dataset

The dataset used in this project is stored in a zip file (`Image_Classification_using_VGG19.zip`), which contains images for training, validation, and testing.

- **Train Set**: The training set contains labeled images used to train the model.
- **Validation Set**: The validation set is used to tune the model during training.
- **Test Set**: The test set is used to evaluate the final performance of the model.

### Dataset Link

You can access and download the dataset from Google Drive using the link below:

[Download Image Classification using VGG-19.zip](https://drive.google.com/drive/folders/1-190aKlLZJWe5PgWBRp2BuHuVBz3Chxg?usp=drive_link)

### Dataset Directory Structure

The dataset is expected to have the following directory structure:

Image Classification using VGG-19/
├── data/
    ├── train/
        ├── class_1/
        ├── class_2/ └── ...
    ├── validation/ 
        ├── class_1/
        ├── class_2/ └── ... 
    └── test/
        ├── class_1/
        ├── class_2/ └── ...

Make sure to extract the zip file and modify the paths accordingly.

## Project Setup

### 1. Mount Google Drive:
The project works in a Google Colab environment, and you need to mount Google Drive to access the dataset and store the trained model.

```python
from google.colab import drive
drive.mount('/content/drive')
```
### 2. Extract the Dataset: 
The dataset is stored in a zip file (Image_Classification_using_VGG19.zip) located in your Google Drive. The zip file is extracted as follows:

```python
zip_file_path = '/content/drive/My Drive/Image_Classification_using_VGG19.zip'
extract_path = '/content/drive/My Drive/Image_Classification_using_VGG19'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```
### 3. Directory Structure: 
After extracting the dataset, ensure the data is located in the proper folder (/data/train, /data/validation, /data/test).

## Model Architecture

The model is based on the VGG19 architecture, with a few modifications for the custom classification task. The VGG19 model is used as a feature extractor (without the fully connected layers) and is followed by:

- **Flatten Layer to flatten the 3D output from the VGG19 base model.**
- **Dense Layer (256 units) with ReLU activation.**
- **Dropout Layer (50%) to prevent overfitting.**
- **Output Layer with softmax activation for multi-class classification.**
The VGG19 model is loaded with pre-trained weights from ImageNet and is fine-tuned by unfreezing the last 4 layers to improve performance on the custom dataset.

## Training the Model

The model is trained using the Adam optimizer with a learning rate of 0.0001. Data augmentation is applied during training to improve generalization, including random rotations, shifts, shearing, zooming, and horizontal flips.

Additionally, class weights are computed to handle class imbalance, ensuring the model doesn't favor more frequent classes.

The model is trained using the following code:
``` bash
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weight_dict,
    callbacks=[lr_reduction]
)
```
The model also uses Learning Rate Scheduling to reduce the learning rate after several epochs without improvement in the validation loss.

## Evaluation and Metrics

After training, the model is evaluated on the test set to calculate the test accuracy. Additionally, various metrics are computed:
- **Confusion Matrix to assess the performance across all classes.**
- **Classification Report including precision, recall, and F1 score for each class.**
- **F1 Score (Weighted) to handle class imbalances.**

Sampl code for evaluation :
```bash
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')
```
Additionally, a confusion matrix is plotted using seaborn to visualize the performance across different classes.

## Results

Once the model is trained and evaluated, the performance metrics such as accuracy, precision, recall, F1 score, and the confusion matrix are printed and visualized. These metrics help determine the effectiveness of the model.




