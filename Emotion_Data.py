import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        
        # Performance tracking
        self.fps = 0
        self.frame_time = 0
        
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
                self.model.load_weights('model.h5')
                print("Loaded pre-trained model")
            except:
                print("No pre-trained model found, initializing new model")

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

    def fine_tune(self, train_dir, validation_dir, epochs=10):
        """Fine-tune the model on custom data"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=32,
            color_mode="grayscale",
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            validation_dir,
            target_size=(48, 48),
            batch_size=32,
            color_mode="grayscale",
            class_mode='categorical'
        )
        
        # Compile and train
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune the model
        self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
        
        # Save the fine-tuned model
        self.model.save_weights('model_fine_tuned.h5')

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