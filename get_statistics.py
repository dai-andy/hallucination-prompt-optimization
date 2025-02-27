import pandas as pd
import numpy as np
import argparse


def calculate_model_accuracies(df, model_name):
    """
    Calculate accuracy statistics for a specific model's predictions compared to actual labels
    """
    # Get the prediction column for the specified model
    prediction_col = f"label_{model_name}"
    if prediction_col not in df.columns:
        raise ValueError(f"Column '{prediction_col}' not found in dataframe")
    
    # Calculate accuracy
    matches = (df[prediction_col] == df['label']).sum()
    total = len(df)
    accuracy = (matches / total) * 100
    
    # Get mismatches
    mismatches = df[df[prediction_col] != df['label']]
    
    return accuracy, mismatches


def print_statistics(accuracy, mismatches, model_name):
    """
    Print accuracy statistics and mismatched predictions for a specific model
    """
    # Print mismatches
    print(f"\n=== Mismatched Predictions ({len(mismatches)} entries) ===")
    if len(mismatches) > 0:
        for idx, row in mismatches.iterrows():
            print(f"Index {idx}:")
            print(f"  Reference: {row['reference'][:100]}...")
            print(f"  Input: {row['input'][:100]}...")
            print(f"  Output: {row['output'][:100]}...")
            print(f"  Actual Label: {row['label']}")
            print(f"  Predicted Label: {row[f'label_{model_name}']}")
            print("-"*50)

    # Print accuracy
    print("\n=== Model Accuracy ===")
    print(f"{model_name}: {accuracy:.2f}%")
    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate accuracy statistics for a specific model')
    parser.add_argument('data_path', type=str, help='Path to the CSV file')
    parser.add_argument('model_name', type=str, help='Name of the model (ie. gpt-4o, gpt-3.5-turbo, claude-3-5-sonnet-latest, or litellm/together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo')
    args = parser.parse_args()
    
    # Read the CSV file
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {args.data_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    try:
        # Calculate statistics
        accuracy, mismatches = calculate_model_accuracies(df, args.model_name)
        
        # Print results
        print_statistics(accuracy, mismatches, args.model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
