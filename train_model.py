from Emotion_Data import EmotionDetector

def main():
    # Initialize a fresh model (don't load pre-trained weights)
    detector = EmotionDetector(use_pretrained=False)

    # Train it using your processed dataset
    detector.fine_tune('data/train', 'data/validation', epochs=10)

if __name__ == "__main__":
    main()
