import os
import shutil
import random

def split_dataset(train_dir='data/train', val_dir='data/validation', split_ratio=0.2):
    for emotion in os.listdir(train_dir):
        emotion_dir = os.path.join(train_dir, emotion)
        val_emotion_dir = os.path.join(val_dir, emotion)
        os.makedirs(val_emotion_dir, exist_ok=True)

        files = os.listdir(emotion_dir)
        random.shuffle(files)
        val_count = int(len(files) * split_ratio)

        for file in files[:val_count]:
            src = os.path.join(emotion_dir, file)
            dst = os.path.join(val_emotion_dir, file)
            shutil.move(src, dst)

if __name__ == "__main__":
    split_dataset()
