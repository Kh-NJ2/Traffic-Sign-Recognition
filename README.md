# Traffic Sign Recognition using a Convolutional Neural Network (CNN)

This project contains the complete pipeline for building, training, and testing a deep learning model to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is built with TensorFlow and Keras.

---

## Features
 
* **High Accuracy**: Achieves over 99% accuracy on the test set.
* **Data Augmentation**: Utilizes data augmentation techniques to improve model robustness and prevent overfitting.
* **Modular Scripts**: Includes separate notebooks/scripts for both training the model from scratch and for running predictions on new images.
* **Official Test Set Evaluation**: Contains code to evaluate the model's performance on the official, held-out test set.

---

## Project Structure

This repository contains the following files:

* `Traffic_sign_classifier.ipynb`: A detailed Jupyter Notebook that walks through the entire process step-by-step, including data loading, preprocessing, model training with data augmentation, evaluation, and testing. **This is the main file for exploring the project.**
* `traffic_sign_model_creation.ipynb`: A simplified notebook focused purely on the model creation, training, and saving process.
* `test_model.py`: A command-line Python script used to run predictions on a single image using the pre-trained model.
* `traffic_sign_model.keras`: The saved, pre-trained Keras model file. You can use this directly for inference without needing to retrain.
* `requirements.txt`: A file listing all the necessary Python packages to run this project.
* `datasets/`: The folder containing the GTSRB dataset (you will need to download this).

---

## Installation

To set up the environment and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Kh-NJ2/Traffic-Sign-Recognition.git
    cd Traffic-Sign-Recognition
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset**:
    * Download the German Traffic Sign Recognition Benchmark (GTSRB) dataset from Kaggle.
    * Unzip the file and place the contents into a folder named `datasets` in the root of the project directory. Your `datasets` folder should contain the `Train` folder, `Test` folder, and `Test.csv`.

---

## Usage

There are two main ways to use this project: exploring the notebook or running the scripts directly.

### 1. Exploring the Jupyter Notebook

For a detailed, step-by-step guide through the entire process, open and run the cells in `Traffic_sign_classifier.ipynb`.

```bash
jupyter notebook Traffic_sign_classifier.ipynb
```

### 2. Running the Prediction Script

To classify a single traffic sign image from your terminal, use the `test_model.py` script. Make sure the `traffic_sign_model.keras` file is in the same directory.

Run the following command, replacing `path/to/your/image.png` with the actual path to your image file:

```bash
python test_model.py --image path/to/your/image.png
```

The script will load the pre-trained model and print the predicted class for your image.

---
