from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "happy"
    elif polarity < -0.2:
        return "sad"
    else:
        return "neutral"

def detect_craving(text):
    craving_keywords = {
        "spicy": ["spicy", "hot", "chili", "pepper"],
        "sweet": ["sweet", "dessert", "cake", "sugar", "chocolate"],
        "cold": ["ice cream", "cold", "frozen", "chill"],
        "healthy": ["salad", "healthy", "green", "fit", "organic"]
    }
    for craving_type, keywords in craving_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return craving_type
    return "general"

def get_recommendations_based_on_sentiment(sentiment):
    recommendations = {
        "happy": ["Sushi", "Tacos", "Smoothies"],
        "sad": ["Chocolate cake", "Comfort food", "Ice Cream"],
        "neutral": ["Pasta", "Fried rice", "Sandwich"]
    }
    return recommendations.get(sentiment, [])

def get_recommendations_based_on_craving(craving):
    recommendations = {
        "spicy": ["Spicy Chicken Wings", "Chili Paneer", "Hot Ramen"],
        "sweet": ["Donuts", "Cupcake", "Chocolate Brownie"],
        "cold": ["Ice Cream", "Frozen Yogurt", "Cold Coffee"],
        "healthy": ["Greek Salad", "Avocado Toast", "Smoothie Bowl"]
    }
    return recommendations.get(craving, [])

def get_sentiment_craving_recommendations(user_input):
    craving = detect_craving(user_input)
    sentiment = analyze_sentiment(user_input)

    if craving != "general":
        return get_recommendations_based_on_craving(craving)
    else:
        return get_recommendations_based_on_sentiment(sentiment)
