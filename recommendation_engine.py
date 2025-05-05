import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path='smart_recommendation_system/cleaned_zomato.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['cuisines'] = df['cuisines'].fillna('')
    df['area'] = df['area'].fillna('')
    df['averagecost'] = df['averagecost'].fillna(df['averagecost'].mean())
    df['isvegonly'] = df['isvegonly'].astype(str)
    df['combined_features'] = df['cuisines'] + ' ' + df['area'] + ' ' + df['isvegonly']
    return df

def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(user_cuisine, user_area, is_veg, df, cosine_sim, top_n=5):
    user_profile = f"{user_cuisine} {user_area} {is_veg}"
    tfidf = TfidfVectorizer(stop_words='english')
    temp_corpus = df['combined_features'].tolist() + [user_profile]
    tfidf_matrix = tfidf.fit_transform(temp_corpus)
    cosine_sim_user = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim_user[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:top_n]]
    return df.iloc[top_indices][['name', 'cuisines', 'area', 'averagecost', 'isvegonly']]

def main():
    df = load_data()
    df = preprocess_data(df)
    cosine_sim = build_similarity_matrix(df)

    print("✅ Model ready for recommendations!")

    cuisine = input("Enter preferred cuisine (e.g., Chinese, North Indian): ")
    area = input("Enter preferred area (e.g., Indiranagar, Koramangala): ")
    is_veg = input("Veg only? (1 for Yes, 0 for No): ")

    recommendations = get_recommendations(cuisine, area, is_veg, df, cosine_sim)
    print("\n🍽️ Top Recommendations for You:")
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
