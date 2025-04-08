import os
import random
import json

class ContentRecommendationEngine:
    """
    A class to handle adaptive content recommendation based on detected emotions and user reactions.
    """

    def __init__(self, media_library_path='media_library.json'):
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

        recommended_video = random.choice(videos)
        return recommended_video

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

    def get_personalized_recommendation(self, user_id, detected_emotion):
        """
        Get a personalized video recommendation based on user preferences and detected emotion.

        Args:
            user_id: The ID of the user.
            detected_emotion: The detected emotion.

        Returns:
            A video URL selected from the relevant category, tailored to user preferences.
        """
        category = self.map_emotion_to_category(detected_emotion)
        videos = self.media_library.get(category, [])

        if not videos:
            return None

        # Adjust recommendations based on user preferences
        if user_id in self.user_preferences and detected_emotion in self.user_preferences[user_id]:
            preferences = self.user_preferences[user_id][detected_emotion]
            like_ratio = preferences["like"] / (preferences["like"] + preferences["dislike"] + 1)

            if random.random() < like_ratio:
                recommended_video = random.choice(videos)
            else:
                other_categories = [cat for cat in self.media_library if cat != category]
                other_videos = [video for cat in other_categories for video in self.media_library[cat]]
                recommended_video = random.choice(other_videos) if other_videos else random.choice(videos)
        else:
            recommended_video = random.choice(videos)

        return recommended_video