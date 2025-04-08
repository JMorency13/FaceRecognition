import os
from content_recommendation_engine import ContentRecommendationEngine

def simulate_feedback(engine, user_id, emotion, feedback):
    # Provide feedback to the recommendation engine
    engine.update_user_preferences(user_id, emotion, feedback)
    print(f"Feedback '{feedback}' for emotion '{emotion}' recorded.")

def test_content_recommendation():
    # Initialize the recommendation engine
    engine = ContentRecommendationEngine()

    # Simulate detected emotions and provide initial recommendations
    emotions = ["Happy", "Sad", "Angry", "Fearful", "Surprised", "Neutral", "Disgusted"]
    user_id = "user123"

    for emotion in emotions:
        recommended_video = engine.get_recommendation(emotion)
        if recommended_video:
            print(f"For emotion '{emotion}', recommended video: {recommended_video}")
        else:
            print(f"No video found for emotion '{emotion}'")

    # Simulate providing feedback
    simulate_feedback(engine, user_id, "Happy", "like")
    simulate_feedback(engine, user_id, "Sad", "dislike")

    # Test personalized recommendations
    for emotion in emotions:
        personalized_video = engine.get_personalized_recommendation(user_id, emotion)
        if personalized_video:
            print(f"For emotion '{emotion}', personalized recommended video: {personalized_video}")
        else:
            print(f"No video found for emotion '{emotion}'")

if __name__ == "__main__":
    test_content_recommendation()