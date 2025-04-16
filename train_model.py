from Emotion_Data import EmotionDetector
from keras.callbacks import ModelCheckpoint, EarlyStopping

def main():
    # Initialize a fresh model (don't load pre-trained weights)
    detector = EmotionDetector(use_pretrained=False)

    # Create callbacks
    checkpoint = ModelCheckpoint(
        'model_checkpoint.h5',        # The file name to save the model
        save_best_only=True,          # Save the model only when the validation loss improves
        monitor='val_loss',           # Monitor validation loss
        mode='min',                   # Minimize validation loss
        verbose=1                      # Print progress
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',          # Monitor validation loss
        patience=10,                 # Number of epochs to wait before stopping if no improvement
        restore_best_weights=True   # Restore the best weights when stopping
    )

    # Train the model with the new callbacks
    detector.fine_tune(
        'data/train',                # Training data directory
        'data/validation',           # Validation data directory
        epochs=100,                  # Total epochs to train
        callbacks=[checkpoint, early_stopping]  # Add checkpoint and early stopping
    )

if __name__ == "__main__":
    main()
