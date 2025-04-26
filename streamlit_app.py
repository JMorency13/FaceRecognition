import streamlit as st
import cv2
import numpy as np
from Emotion_Data import EmotionDetector
import os
import time
import random

# Maps emotions to folder names
emotion_to_folder = {
    "Happy": "comedy",
    "Sad": "nature",
    "Angry": "animals",
    "Fearful": "nature",
    "Surprised": "comedy",
    "Neutral": "comedy",
    "Disgusted": "animals"
}

def get_local_video(emotion):
    if emotion not in emotion_to_folder:
        return None
    folder = emotion_to_folder[emotion]
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.exists(folder_path):
        return None
    videos = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.mov', '.avi'))]
    return os.path.join(folder, random.choice(videos)) if videos else None

def main():
    st.set_page_config(layout="wide", page_title="Emotion-Based Video Player")

    if 'emotion_detector' not in st.session_state:
        st.session_state.emotion_detector = EmotionDetector()

    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = None

    if 'last_video_update' not in st.session_state:
        st.session_state.last_video_update = time.time()

    if 'recommended_videos' not in st.session_state:
        st.session_state.recommended_videos = []

    if 'video_recommendation_stopped' not in st.session_state:
        st.session_state.video_recommendation_stopped = False

    # Styling
    st.markdown("""
        <style>
        .emotion-box {
            padding: 10px;
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }
        .webcam-container, .video-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)

        if st.session_state.video_recommendation_stopped:
            st.subheader("😊 Mood Improved!")
            st.write("Here is a summary of your video journey:")

            for rec in st.session_state.recommended_videos:
                emotion = rec['emotion']
                video = rec['video_path']
                improved = rec['led_to_happy']
                st.write(f"🎬 **{video}** | Emotion: *{emotion}* ➜ {'😊 Mood Improved' if improved else '😐 No Change'}")
        else:
            video_path = get_local_video(st.session_state.current_emotion)
            if video_path:
                st.video(video_path)
                st.write("Detected Emotion:", st.session_state.current_emotion)
                st.write("Video Path:", video_path)

                # Append new video recommendation if it's a new emotion
                if not st.session_state.recommended_videos or st.session_state.recommended_videos[-1]['emotion'] != st.session_state.current_emotion:
                    st.session_state.recommended_videos.append({
                        'emotion': st.session_state.current_emotion,
                        'video_path': video_path,
                        'led_to_happy': False
                    })
            else:
                st.warning("No video found for this emotion.")
                st.info("Waiting for emotion to be detected...")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)

        if not st.session_state.video_recommendation_stopped:
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break

                processed_frame = st.session_state.emotion_detector.process_frame(frame)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

                new_emotion = st.session_state.emotion_detector.current_emotion
                current_time = time.time()
                cooldown_duration = 10  # seconds

                if new_emotion != st.session_state.current_emotion and (current_time - st.session_state.last_video_update >= cooldown_duration):
                    st.session_state.current_emotion = new_emotion
                    st.session_state.last_video_update = current_time

                    # If Happy is detected, mark previous video as successful
                    if new_emotion == "Happy":
                        if st.session_state.recommended_videos:
                            st.session_state.recommended_videos[-1]['led_to_happy'] = True
                        st.session_state.video_recommendation_stopped = True
                        cap.release()
                        st.rerun()
                        break
                    else:
                        st.rerun()

                time.sleep(0.1)

            cap.release()

        else:
            st.info("Emotion detection has been stopped because the user is happy! 🥳")
        

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
