import pandas as pd

def normalize_data(dataset_path):
    # Read the dataset from the given path
    df = pd.read_csv(dataset_path)
    
    # Normalize the RGB values to have a mean of 0 and standard deviation of 1
    df[['Red', 'Green', 'Blue']] = (df[['Red', 'Green', 'Blue']] - df[['Red', 'Green', 'Blue']].mean()) / df[['Red', 'Green', 'Blue']].std()
    
    # Save the updated dataset to a new CSV file
    updated_dataset_path = "./data/fruit_to_emoji/playground/normalized_fruit_data.csv"
    df.to_csv(updated_dataset_path, index=False)
    
    return updated_dataset_path

# Call the function with the provided dataset path
updated_dataset_path = normalize_data("./data/fruit_to_emoji/playground/fruit_data.csv")