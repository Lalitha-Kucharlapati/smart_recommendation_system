import pandas as pd

def clean_zomato_data(file_path='smart_recommendation_system/data/zomato.csv', save_path='cleaned_zomato.csv'):
    # Load dataset
    df = pd.read_csv(file_path, encoding='latin-1')

    # Drop irrelevant or extremely sparse columns
    df = df.drop(columns=['URL', 'Full_Address', 'PhoneNumber', 'KnownFor'])

    # Fill missing values
    df['PopularDishes'] = df['PopularDishes'].fillna("Not Specified")
    df['PeopleKnownFor'] = df['PeopleKnownFor'].fillna("Not Specified")
    df['Timing'] = df['Timing'].fillna("Timing Not Available")

    # Drop rows with missing Cuisines or Area (core features)
    df = df.dropna(subset=['Cuisines', 'Area'])

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Convert AverageCost to numeric
    df['averagecost'] = pd.to_numeric(df['averagecost'], errors='coerce')

    # Drop rows with missing cost
    df = df.dropna(subset=['averagecost'])

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned version
    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned data saved to '{save_path}' with shape: {df.shape}")

if __name__ == "__main__":
    clean_zomato_data()
