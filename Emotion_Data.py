import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
import os
from sklearn.utils.class_weight import compute_class_weight
import time
import random
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ContentRecommendationEngine:
    """
    A class to handle adaptive content recommendation based on detected emotions and user reactions.
    """

    def __init__(self, media_library_path='project/media_library.json'):
        """
        Initialize the recommendation engine with:
        - Media library containing categorized videos
        """
        self.media_library_path = media_library_path
        self.media_library = self.load_media_library()
        self.user_preferences = {}

    def load_media_library(self):
        """
        Load the media library from a JSON file containing categorized videos.

        Returns:
            A dictionary with categories as keys and lists of video URLs as values.
        """
        if not os.path.exists(self.media_library_path):
            raise FileNotFoundError(f"Media library file not found: {self.media_library_path}")

        with open(self.media_library_path, 'r') as file:
            media_library = json.load(file)

        return media_library

    def get_recommendation(self, detected_emotion):
        """
        Get a video recommendation based on the detected emotion.

        Args:
            detected_emotion: The detected emotion (e.g., "Happy", "Sad").

        Returns:
            A video URL selected from the relevant category.
        """
        category = self.map_emotion_to_category(detected_emotion)
        videos = self.media_library.get(category, [])

        if not videos:
            return None

        return random.choice(videos)

    def map_emotion_to_category(self, emotion):
        """
        Map detected emotions to video categories.

        Args:
            emotion: The detected emotion (e.g., "Happy", "Sad").

        Returns:
            The corresponding category (e.g., "comedy", "nature").
        """
        emotion_category_map = {
            "Happy": "comedy",
            "Sad": "nature",
            "Angry": "animals",
            "Fearful": "nature",
            "Surprised": "comedy",
            "Neutral": "nature",
            "Disgusted": "animals"
        }
        return emotion_category_map.get(emotion, "nature")

    def update_user_preferences(self, user_id, emotion, feedback):
        """
        Update user preferences based on feedback.

        Args:
            user_id: The ID of the user.
            emotion: The detected emotion.
            feedback: The feedback provided by the user (e.g., "like", "dislike").
        """
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}

        if emotion not in self.user_preferences[user_id]:
            self.user_preferences[user_id][emotion] = {"like": 0, "dislike": 0}

        self.user_preferences[user_id][emotion][feedback] += 1


class EmotionDetector:
    """
    A class to handle real-time emotion detection using a custom CNN model.
    """

    def __init__(self, use_pretrained=True):
        """
        Initialize the emotion detector with necessary components:
        - CNN Model
        - Video capture device
        - Face detection cascade
        - Performance metrics
        """
        # Initialize webcam
        self.video_capture = cv2.VideoCapture(0)
        
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion dictionary
        self.emotion_dict = {
            0: "Angry", 
            1: "Disgusted", 
            2: "Fearful", 
            3: "Happy", 
            4: "Neutral", 
            5: "Sad", 
            6: "Surprised"
        }
        
        # Dictionary to store emotion colors for visualization
        self.emotion_colors = {
            "Happy": (0, 255, 0),     # Green
            "Sad": (0, 0, 255),       # Red
            "Angry": (0, 0, 255),     # Red
            "Fearful": (255, 0, 0),   # Blue
            "Surprised": (255, 255, 0),# Yellow
            "Neutral": (255, 255, 255),# White
            "Disgusted": (128, 0, 128) # Purple
        }

        # Initialize and load the model
        self.model = self.create_model()
        if use_pretrained:
            try:
                self.model.load_weights('model_fine_tuned.h5')
                print("Loaded pre-trained model")
            except:
                print("No pre-trained model found, initializing new model")

        # Initialize content recommendation engine
        self.recommendation_engine = ContentRecommendationEngine()
        self.user_id = "user123"  # Example user ID
        self.current_emotion = None
        self.start_time = time.time()
        # Store recommended videos and feedback
        self.recommended_videos = {emotion: [] for emotion in self.emotion_dict.values()}
        self.recommendation_active = True

    def create_model(self):
        """Create and return the CNN model architecture"""
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
        """
        Process a single frame and detect emotions.
        
        Args:
            frame: Video frame to process
            
        Returns:
            Processed frame with emotion annotations
        """
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        
        # Process each face detected in the frame
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            
            # Extract and preprocess the face region
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
            )
            
            # Predict emotion
            prediction = self.model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            emotion = self.emotion_dict[maxindex]
            
            # Draw emotion label
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.putText(frame, emotion, (x+20, y-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Handle video recommendation every 20 seconds
            if self.recommendation_active and (time.time() - self.start_time) >= 20:
                self.recommendation_active = False
                recommended_video = self.recommendation_engine.get_recommendation(emotion)
                if recommended_video:
                    print(f"\nFor emotion '{emotion}', recommended video: {recommended_video}")
                    feedback = input("Did this video improve your mood? (yes/no): ").strip().lower()
                    feedback_status = "Yes" if feedback == "yes" else "No"
                    self.recommended_videos[emotion].append((recommended_video, feedback_status))
                    self.recommendation_engine.update_user_preferences(
                        self.user_id, emotion, "like" if feedback == "yes" else "dislike"
                    )
                self.start_time = time.time()
                self.recommendation_active = True

        return frame

    def run(self):
        """
        Main loop to capture and process webcam feed
        """
        try:
            while True:
                # Capture frame from webcam
                ret, frame = self.video_capture.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow('Emotion Detection', 
                          cv2.resize(processed_frame, (1600,960),
                                     interpolation = cv2.INTER_CUBIC))
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.video_capture.release()
            cv2.destroyAllWindows()
            self.display_recommendation_summary()

    def display_recommendation_summary(self):
        """
        Display a summary of recommended videos based on detected emotions and user feedback
        """
        print("\nRecommendation Summary:")
        for emotion, video_feedback_list in self.recommended_videos.items():
            print(f"\nEmotion: {emotion}")
            for video, feedback in video_feedback_list:
                print(f"  - Video: {video}, Liked: {feedback}")

def main():
    """
    Main function to initialize and run the emotion detector
    """
    try:
        detector = EmotionDetector()
        detector.run()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
