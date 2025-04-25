import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
import os
import time
import random
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ContentRecommendationEngine:
    def __init__(self, media_library_path='media_library.json', base_path='project/'):
        self.media_library_path = media_library_path
        self.base_path = base_path  # Base path for video files
        self.media_library = self.load_media_library()
        self.user_preferences = {}

    def load_media_library(self):
        if not os.path.exists(self.media_library_path):
            raise FileNotFoundError(f"Media library file not found: {self.media_library_path}")
        with open(self.media_library_path, 'r') as file:
            return json.load(file)

    def get_recommendation(self, detected_emotion):
        category = self.map_emotion_to_category(detected_emotion)
        videos = self.media_library.get(category, [])
        if videos:
            # Prepend the base path to the selected video file
            return os.path.join(self.base_path, random.choice(videos))
        return None

    def map_emotion_to_category(self, emotion):
        emotion_category_map = {
            "Happy": "comedy",
            "Sad": "nature",
            "Angry": "animals",
            "Fearful": "nature",
            "Neutral": "comedy",
        }
        return emotion_category_map.get(emotion, "nature")

    def update_user_preferences(self, user_id, emotion, feedback):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        if emotion not in self.user_preferences[user_id]:
            self.user_preferences[user_id][emotion] = {"like": 0, "dislike": 0}
        self.user_preferences[user_id][emotion][feedback] += 1

class EmotionDetector:
    def __init__(self, use_pretrained=True):
        self.video_capture = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_dict = {
            0: "Angry", 
            2: "Fearful", 
            3: "Happy", 
            4: "Neutral", 
            5: "Sad", 
        }
        self.emotion_colors = {
            "Happy": (0, 255, 0),
            "Sad": (0, 0, 255),
            "Angry": (0, 0, 255),
            "Fearful": (255, 0, 0),
            "Neutral": (255, 255, 255),
        }
        self.model = self.create_model()
        if use_pretrained:
            try:
                self.model.load_weights('model_fine_tuned.h5')
                print("Loaded pre-trained model")
            except:
                print("No pre-trained model found, initializing new model")
        self.recommendation_engine = ContentRecommendationEngine()
        self.user_id = "user123"
        self.current_emotion = None
        self.start_time = time.time()
        self.recommended_videos = {emotion: [] for emotion in self.emotion_dict.values()}

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        return model

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            prediction = self.model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]
            self.current_emotion = emotion  # <-- THIS LINE IS CRUCIAL
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.putText(frame, emotion, (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 10:
                recommended_video = self.recommendation_engine.get_recommendation(emotion)
                if recommended_video:
                    print(f"For emotion '{emotion}', recommended video: {recommended_video}")
                self.start_time = time.time()

        return frame

    def run(self):
        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                processed_frame = self.process_frame(frame)
                cv2.imshow('Emotion Detection', 
                           cv2.resize(processed_frame, (1600,960), interpolation=cv2.INTER_CUBIC))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.video_capture.release()
            cv2.destroyAllWindows()

def main():
    try:
        detector = EmotionDetector()
        detector.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
