import streamlit as st
import pandas as pd
from recommendation_engine import (
    load_data, preprocess_data,
    build_similarity_matrix, get_recommendations
)
from sentiment_craving_module import get_sentiment_craving_recommendations

# Load and preprocess data
df = preprocess_data(load_data("smart_recommendation_system/cleaned_zomato.csv"))
cosine_sim = build_similarity_matrix(df)

st.set_page_config(page_title="Smart Food Recommendation System", layout="centered")

st.title("🍽️ Smart Food Recommendation System")
st.write("Get personalized restaurant suggestions based on your preferences or cravings!")

# Sidebar inputs
st.sidebar.header("Your Preferences")
cuisine = st.sidebar.text_input("Preferred Cuisine")
area = st.sidebar.text_input("Preferred Area")
veg_option = st.sidebar.radio("Veg Only?", options=["Yes", "No"])
is_veg = "1" if veg_option == "Yes" else "0"

use_sentiment = st.sidebar.checkbox("Use Mood/Craving Based Suggestion")

# Mood/craving input (only visible when checkbox is ticked)
user_input = ""
if use_sentiment:
    user_input = st.text_input("What's your current mood or craving? (e.g., I want something sweet)", key="craving_input")

# Button to trigger recommendation
if st.sidebar.button("Get Recommendations"):
    if use_sentiment:
        if user_input.strip():
            # Get dish suggestions
            rec_dishes = get_sentiment_craving_recommendations(user_input)

            st.subheader("🎯 Craving-Based Dish Suggestions")
            for i, dish in enumerate(rec_dishes, start=1):
                st.markdown(f"**{i}. {dish}**")

            # Search for restaurants with matching dishes/cuisines
            matches = pd.DataFrame()
            for dish in rec_dishes:
                filtered = df[df['cuisines'].str.contains(dish.split()[0], case=False, na=False)]
                matches = pd.concat([matches, filtered])

            if not matches.empty:
                matches = matches.drop_duplicates().reset_index(drop=True)
                st.subheader("🏪 Restaurants That May Serve These")
                st.dataframe(matches[['name', 'area', 'cuisines', 'averagecost', 'isvegonly']])
            else:
                st.info("No exact restaurant matches found for these cravings.")
        else:
            st.warning("Please enter your mood or craving.")
    else:
        if cuisine and area:
            recs = get_recommendations(cuisine, area, is_veg, df, cosine_sim)
            st.subheader("🍴 Recommended Restaurants for You")
            st.dataframe(recs.reset_index(drop=True))
        else:
            st.warning("Please fill in both cuisine and area.")
