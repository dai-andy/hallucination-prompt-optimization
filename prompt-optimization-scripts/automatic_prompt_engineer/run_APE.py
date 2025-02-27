import openai
from automatic_prompt_engineer import ape
import os
import pandas as pd
import argparse


def prepare_dataset(df):
    """Convert dataframe into format expected by APE"""
    # Format inputs as strings
    inputs = []
    for _, row in df.iterrows():
        formatted_input = f"""Reference: {row['reference']}
Query: {row['input']}
Answer: {row['output']}"""
        inputs.append(formatted_input)
    
    # Get outputs
    outputs = df['label'].tolist()
    
    data = (inputs, outputs)
    
    return data


def create_demos_template():
    """Create template for demonstrations"""
    return """Reference: [INPUT_0]
Query: [INPUT_1]
Answer: [INPUT_2]
Label: [OUTPUT]"""


def run_ape(train_data, input_template_path):
    """Run APE to optimize the prompt template"""
    # Load the initial prompt template
    with open(input_template_path, 'r') as f:
        eval_template = f.read().strip()
    
    # Create demos template
    demos_template = create_demos_template()
    
    # Configure APE
    config = {
        'eval_model': 'gpt-3.5-turbo',
        'prompt_gen_model': 'gpt-3.5-turbo',
        'num_prompts': 10,  # Number of candidate prompts to generate
        'num_eval_samples': 10,  # Number of samples to use for evaluation
        'num_few_shot': 2,  # Number of few-shot examples to use
    }
    
    # # Run APE
    # cost = ape.simple_estimate_cost(
    #     dataset=train_data,  # Already in (inputs, outputs) format
    #     eval_template=eval_template,
    #     demos_template=demos_template,
    #     eval_model=config['eval_model'],
    #     prompt_gen_model=config['prompt_gen_model'],
    #     num_prompts=config['num_prompts'],
    #     eval_rounds=3,
    #     prompt_gen_batch_size=5,
    #     eval_batch_size=10,
    # )
    # print(f'Estimated cost of {cost}')

    results, demo_fn = ape.simple_ape(
        dataset=train_data,
        eval_template=eval_template,
        demos_template=demos_template,
        eval_model=config['eval_model'],
        prompt_gen_model=config['prompt_gen_model'],
        num_prompts=config['num_prompts'],
        eval_rounds=3,
        prompt_gen_batch_size=5,
        eval_batch_size=10
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Optimize prompt template using APE')
    parser.add_argument('data_path', type=str, help='Path to the input CSV file')
    parser.add_argument('template_path', type=str, help='Path to the input prompt template file')
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    
    # Read input data
    try:
        df = pd.read_csv(args.data_path)
        required_columns = ['reference', 'input', 'output', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Prepare dataset
    train_data = prepare_dataset(df)
    
    # Run APE
    results = run_ape(train_data, args.template_path)
    
    print(f"Results are {results}")


if __name__ == "__main__":
    main()