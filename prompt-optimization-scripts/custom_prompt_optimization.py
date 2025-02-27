import openai
import pandas as pd
from tqdm import tqdm
import json
import time
import argparse
import os


def get_hallucination_detection(query, reference, response, prompt_template, model_name, api_key):
    """Get hallucination detection result using current prompt"""
    client = openai.OpenAI(api_key=api_key)
    
    prompt = prompt_template.format(
        query=query,
        reference=reference,
        response=response
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always responds in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        
        result = json.loads(response.choices[0].message.content)
        if not isinstance(result, dict) or 'label' not in result:
            return {
                'label': 'error',
                'explanation': 'Invalid response format'
            }
        return result
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return {
            'label': 'error',
            'explanation': f'API error: {str(e)}'
        }


def get_critic_feedback(query, reference, response, actual_label, detection_result, model_name, api_key):
    """Critic agent analyzes hallucination detection result"""
    client = openai.OpenAI(api_key=api_key)
    
    critic_prompt = f"""Analyze this hallucination detection result and provide critical feedback:

Input:
- Query: {query}
- Reference: {reference}
- Response: {response}
- Actual Label: {actual_label}

Hallucination Detector Output:
- Predicted Label: {detection_result['label']}
- Explanation: {detection_result['explanation']}

Provide a critical analysis of the Hallucination Detector's output. Consider:
1. Whether the prediction was correct and why
2. How well the explanation justifies the prediction
3. Any contextual nuances that were missed (ie. contextual overextension)
4. Suggestions for improving the detection process

Provide your analysis in clear, structured text format.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a senior AI safety auditor who provides detailed analysis of hallucination detection results."},
                {"role": "user", "content": critic_prompt}
            ],
            temperature=0,
        )
        
        feedback = response.choices[0].message.content.strip()
        if not feedback:
            raise ValueError("Empty response received")
            
        print('-'*50)
        print(f'Critic feedback:\n{feedback}')
        print('-'*50)
        
        return {
            'feedback': feedback,
            'correct_prediction': detection_result['label'] == actual_label
        }
        
    except Exception as e:
        print(f"Error in critic feedback: {str(e)}")
        print(f"Raw response: {response.choices[0].message.content if response else 'No response'}")
        return {
            'feedback': f'Error in generating feedback: {str(e)}',
            'correct_prediction': False
        }


def get_prompt_update(current_prompt, critic_feedback_list, model_name, api_key):
    """Prompt updater agent suggests prompt improvements based on critic feedback"""
    client = openai.OpenAI(api_key=api_key)
    
    # Combine all feedback into a single string
    all_feedback = "\n\n".join([f"Case {i+1}:\n{feedback['feedback']}" 
                               for i, feedback in enumerate(critic_feedback_list)])
    
    updater_prompt = f"""Based on the critic's feedback, improve this hallucination detection prompt:

Current Prompt:
{current_prompt}

Critic's Feedback:
{all_feedback}

Provide an improved version of the prompt that addresses the critic's feedback. Focus on:
1. Fixing any issues identified by the critic
2. Making the instructions more precise
3. Improving the explanation requirements
4. Maintaining the template of taking in query, reference text, and answer as inputs
5. Maintaining clear JSON output format

Return only the improved prompt text without any additional commentary."""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer who improves prompts based on feedback."},
            {"role": "user", "content": updater_prompt}
        ],
        temperature=0,
    )
    
    print(f'Updated prompt is {response.choices[0].message.content.strip()}')
    print('-'*50)
    return response.choices[0].message.content.strip()


def evaluate_prompt(prompt, test_data, model_name, api_key):
    """Evaluate prompt performance on test data"""
    correct = 0
    total = len(test_data)
    
    for _, row in test_data.iterrows():
        result = get_hallucination_detection(
            row['input'], 
            row['reference'], 
            row['output'], 
            prompt,
            model_name,
            api_key
        )
        if result['label'] == row['label']:
            correct += 1
            
    return correct / total


def select_evaluation_data(df, model_name, max_size=20):
    """Select evaluation data based on prediction mismatches and correct predictions"""
    # Find mismatches
    mismatches = df[df[f'label_{model_name}'] != df['label']]
    n_mismatches = len(mismatches)
    
    if n_mismatches == 0:
        print("No mismatches found - model is performing perfectly!")
        return None
        
    # Limit size
    n_samples = min(max_size, n_mismatches)
    mismatches = mismatches.sample(n=n_samples, random_state=24)
    
    # Find correct predictions
    correct_predictions = df[df[f'label_{model_name}'] == df['label']]
    
    # Sample same number of correct predictions as mismatches
    sampled_correct = correct_predictions.sample(n=n_samples, random_state=24)
    
    # Combine mismatches and sampled correct predictions
    eval_df = pd.concat([mismatches, sampled_correct])
    
    print(f"Selected {len(eval_df)} evaluation examples ({n_samples} mismatches, {n_samples} correct predictions)")
    print(f"Original dataset had {n_mismatches} total mismatches")
    
    return eval_df


def save_evaluation_data(eval_df, output_path):
    """Save evaluation data in txt format with each row as 'text  label'"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in eval_df.iterrows():
            text = f"Reference: {row['reference']} Query: {row['input']} Response: {row['output']}"

            label = '1' if row['label'] == 'hallucinated' else '0'

            f.write(f"{text}\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description='Iteratively refine hallucination detection prompt')
    parser.add_argument('data_path', help='Path to base prompt results CSV')
    parser.add_argument('initial_prompt_path', help='Path to initial prompt template')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--base_model', default='gpt-3.5-turbo', help='Model name in base results')
    parser.add_argument('--iterations', type=int, default=5, help='Number of refinement iterations')
    parser.add_argument('--output_dir', default='templates', help='Directory to save refined prompts')
    args = parser.parse_args()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
        
    # Load data and initial prompt
    df = pd.read_csv(args.data_path)
    with open(args.initial_prompt_path) as f:
        current_prompt = f.read()
    
    # Select evaluation data based on base model results
    eval_df = select_evaluation_data(df, args.base_model)
    if eval_df is None:
        return
        
    print(f"Evaluation set size: {len(eval_df)} ({len(eval_df)//2} mismatches, {len(eval_df)//2} correct predictions)")
    
    # Save evaluation data
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # eval_data_path = f"evaluation_data_{timestamp}.txt"
    # save_evaluation_data(eval_df, eval_data_path)
    # print(f"Saved evaluation data to {eval_data_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initial evaluation
    initial_accuracy = evaluate_prompt(current_prompt, eval_df, args.model, api_key)
    print(f"Initial prompt accuracy: {initial_accuracy:.2%}")
    
    # Refinement iterations
    for iteration in range(args.iterations):
        print(f"\nIteration {iteration + 1}/{args.iterations}")
        
        # Sample cases for analysis
        sample_cases = eval_df.sample(n=min(5, len(eval_df)))
        all_feedback = []
        
        # Get critic feedback for sample cases
        for _, case in tqdm(sample_cases.iterrows()):
            detection_result = get_hallucination_detection(
                case['input'],
                case['reference'],
                case['output'],
                current_prompt,
                args.model,
                api_key
            )
            
            feedback = get_critic_feedback(
                case['input'],
                case['reference'],
                case['output'],
                case['label'],
                detection_result,
                args.model,
                api_key
            )
            all_feedback.append(feedback)
            
        # Update prompt based on accumulated feedback
        updated_prompt = get_prompt_update(current_prompt, all_feedback, args.model, api_key)
        
        # Evaluate updated prompt
        new_accuracy = evaluate_prompt(updated_prompt, eval_df, args.model, api_key)
        print(f"Updated prompt accuracy: {new_accuracy:.2%}")
        
        # Check if we've achieved perfect accuracy
        if new_accuracy == 1.0:
            print("\nAchieved perfect accuracy!")
            # Save final prompt with timestamp
            filename = f"prompt_final_{timestamp}.txt"
            with open(f"{args.output_dir}/{filename}", 'w') as f:
                f.write(updated_prompt)
            print(f"Saved final prompt to {args.output_dir}/{filename}")
            break
            
        # Save if improved
        if new_accuracy > initial_accuracy:
            current_prompt = updated_prompt
            initial_accuracy = new_accuracy
            
            # Save improved prompt with timestamp
            filename = f"prompt_iteration_{iteration + 1}_{timestamp}.txt"
            with open(f"{args.output_dir}/{filename}", 'w') as f:
                f.write(updated_prompt)
            print(f"Saved improved prompt to {args.output_dir}/{filename}")
        else:
            print("No improvement in accuracy, keeping previous prompt")


if __name__ == "__main__":
    main() 