import numpy as np
import cv2
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras


def preprocess_image(image_path):

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return None

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image file '{image_path}'.")
        return None

    image_resized = cv2.resize(image, (32, 32))
    image_normalized = image_resized.astype('float32') / 255.0
    
    image_reshaped = np.reshape(image_normalized, (1, 32, 32, 3))
    
    return image_reshaped

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify a traffic sign image using a trained Keras model.')
    parser.add_argument('-i', '--image', required=True, help='Path to the input image file you want to classify.')
    args = parser.parse_args()

    MODEL_PATH = 'traffic_sign_model.keras'

    if not os.path.exists(MODEL_PATH):
        print(f"Error: The model file '{MODEL_PATH}' was not found. Make sure it's in the same folder as the script.")
    else:
        print("Loading the trained model...")
      
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")

        preprocessed_image = preprocess_image(args.image)

        if preprocessed_image is not None:
            print(f"\nMaking a prediction for: {args.image}")

            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]

            print("\n--- RESULT ---")
            print(f"The model predicts this sign is: Class {predicted_class}")