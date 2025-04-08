import pandas as pd
import numpy as np
import cv2
import os

def load_fer2013(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Emotion mapping
    emotion_mapping = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised"
    }
    
    # Create directories for each emotion
    for emotion in emotion_mapping.values():
        os.makedirs(f'data/train/{emotion}', exist_ok=True)
    
    # Process each row in the CSV
    for index, row in data.iterrows():
        # Get the pixel values and emotion
        pixels = row['pixels'].split(' ')
        emotion = emotion_mapping[row['emotion']]  # Map integer to emotion name
        
        # Convert pixel values to a numpy array
        img_array = np.array(pixels, dtype='float32')
        img_array = img_array.reshape(48, 48)  # Reshape to 48x48
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
        
        # Save the image
        img_path = f'data/train/{emotion}/{index}.png'
        cv2.imwrite(img_path, img_array)

if __name__ == "__main__":
    load_fer2013('fer2013.csv') 