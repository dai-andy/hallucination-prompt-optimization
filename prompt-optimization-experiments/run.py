import pandas as pd
import openai
import time
from tqdm import tqdm
import argparse
import os
import json


def load_template(template_path):
    """Load prompt template from file"""
    try:
        with open(template_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Error loading template from {template_path}: {e}")


def create_prompt(query, reference, response, template):
    """Create prompt using template from file"""
    return template.format(
        query=query,
        reference=reference,
        response=response
    )


def get_openai_response(prompt, model_name, api_key, max_retries=3):
    """Make API call to OpenAI with retry logic"""
    for attempt in range(max_retries):
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always responds in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            # Validate response format
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            
            label = result.get('label', '').strip().lower()
            explanation = result.get('explanation', '').strip()
            
            if label not in ['factual', 'hallucinated']:
                print(f"Warning: Unexpected label format: {label}")
                label = 'error'
                explanation = 'Invalid label format'
            
            return label, explanation
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response_text}")
            return 'error', f'Failed to parse JSON response: {str(e)}'
        except Exception as e:
            print(f"Error processing response: {response_text}")
            return 'error', f'Failed to process response: {str(e)}'
            

def process_dataframe(df, model_name, api_key, prompt_template, batch_size=5):
    """Process the dataframe and add predictions"""
    openai.api_key = api_key
    
    # Initialize new column
    df[f'label_{model_name}'] = None
    
    # Process rows with progress bar
    for idx in tqdm(range(len(df))):
        prompt = create_prompt(
            query=df.iloc[idx]['input'],
            reference=df.iloc[idx]['reference'],
            response=df.iloc[idx]['output'],
            template=prompt_template
        )
        
        label, explanation = get_openai_response(prompt, model_name, api_key)
        df.at[idx, f'label_{model_name}'] = label
        df.at[idx, f'explanation_{model_name}'] = explanation
        
        # Save progress periodically
        if (idx + 1) % batch_size == 0:
            df.to_csv('predictions_backup.csv', index=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate predictions using OpenAI API')
    parser.add_argument('data_path', type=str, help='Path to the input CSV file')
    parser.add_argument('model_name', type=str, help='OpenAI model name (e.g., gpt-4)')
    parser.add_argument('output_path', type=str, help='Path to save the output CSV file')
    parser.add_argument('prompt_template_path', type=str, help='Path to the prompt template file')
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load template
    try:
        prompt_template = load_template(args.prompt_template_path)
    except ValueError as e:
        print(e)
        return
    
    # Read input data
    try:
        df = pd.read_csv(args.data_path)
        required_columns = ['reference', 'input', 'output']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Process the dataframe
    try:
        df = process_dataframe(df, args.model_name, api_key, prompt_template)
        df.to_csv(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
        # Save whatever progress we made
        df.to_csv('predictions_error_backup.csv', index=False)


if __name__ == "__main__":
    main() 