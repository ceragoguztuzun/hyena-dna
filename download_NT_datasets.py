from datasets import load_dataset
import pandas as pd
import os

# Load the dataset
dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")

# Get unique task names
tasks = set(dataset['train']['task'])

# Create directory structure and save as CSV
os.makedirs('data/nucleotide_transformer', exist_ok=True)

for task in tasks:
    # Filter for this task
    train_data = dataset['train'].filter(lambda x: x['task'] == task)
    test_data = dataset['test'].filter(lambda x: x['task'] == task)
    
    # Convert to pandas and save
    task_dir = f'data/nucleotide_transformer/{task}'
    os.makedirs(task_dir, exist_ok=True)
    
    pd.DataFrame(train_data).to_csv(f'{task_dir}/train.csv', index=False)
    pd.DataFrame(test_data).to_csv(f'{task_dir}/test.csv', index=False)
    
    print(f"Saved {task} dataset")