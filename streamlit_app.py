import streamlit as st
import cv2
import numpy as np
from Emotion_Data import EmotionDetector, ContentRecommendationEngine
import time

def get_video_for_emotion(emotion):
    # Video mapping - customize these for your content
    video_map = {
        "Angry": "https://www.youtube.com/watch?v=SpXw0qiy3Wo",  # Funny videos to cheer up
        "Sad": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",    # Happy music
        "Fearful": "https://www.youtube.com/watch?v=W3P7LH-z6gw", # Calming content
        "Happy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",   # Keep happy
        "Neutral": "https://www.youtube.com/watch?v=W3P7LH-z6gw",
        "Surprised": "https://www.youtube.com/watch?v=SpXw0qiy3Wo",
        "Disgusted": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
    }
    return video_map.get(emotion, video_map["Happy"])  # Default to Happy video

def main():
    st.set_page_config(layout="wide", page_title="Emotion-Based Video Player")

    # Initialize emotion detector
    if 'emotion_detector' not in st.session_state:
        st.session_state.emotion_detector = EmotionDetector()
        
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = "Neutral"
        
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #f0f2f6;
            color: black;
            font-size: 20px;
            padding: 10px 30px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            margin: 0 auto;
            display: block;
        }
        .emotion-box {
            padding: 10px;
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
            text-align: center;
            margin: 10px 0;
        }
        .video-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .webcam-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns with custom widths
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        # Display video based on emotion
        current_video = get_video_for_emotion(st.session_state.current_emotion)
        st.video(current_video)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Capture frame
            ret, frame = cap.read()
            if ret:
                # Process frame for emotion detection
                processed_frame = st.session_state.emotion_detector.process_frame(frame)
                
                # Convert BGR to RGB
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                st.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                
                # Update emotion every 20 seconds
                current_time = time.time()
                if current_time - st.session_state.last_update_time >= 20:
                    st.session_state.current_emotion = st.session_state.emotion_detector.current_emotion
                    st.session_state.last_update_time = current_time
                    st.rerun()
            else:
                st.error("Failed to capture frame from webcam")
        else:
            st.error("Failed to access webcam")
            
        cap.release()
        
        # Display detected emotion
        st.markdown(f"""
            <div class="emotion-box">
                Current Emotion: {st.session_state.current_emotion}
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Centered button container
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Change Video"):
        st.rerun()

if __name__ == "__main__":
    main() 