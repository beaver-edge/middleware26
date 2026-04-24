import pandas as pd

# Read the dataset from the given path
df = pd.read_csv("./data/fruit_to_emoji/playground/normalized_fruit_data.csv")

# One-hot encode the 'Fruit' column
df = pd.get_dummies(df, columns=['Fruit'])

# Save the updated dataset to a new CSV file
updated_dataset_path = "./data/fruit_to_emoji/playground/one_hot_encoded_fruit_data.csv"
df.to_csv(updated_dataset_path, index=False)