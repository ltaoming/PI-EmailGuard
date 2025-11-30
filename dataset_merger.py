#!/usr/bin/env python3
"""
Dataset merger for PI-EmailGuard project
Merges microsoft/llmail-inject-challenge and snoop2head/enron_aeslc_emails datasets
"""

from datasets import load_dataset, Dataset, concatenate_datasets
import json
import re

def format_enron_email_to_output_format(email_text):
    """
    Convert Enron email format to the output format used in the first dataset
    """
    # Parse the email components
    lines = email_text.strip().split('\n')
    
    subject = ""
    body = ""
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Subject:'):
            subject = line.replace('Subject:', '').strip()
        elif line.startswith('Body:'):
            current_section = 'body'
            body_start = line.find('Body:') + 5
            if body_start < len(line):
                body = line[body_start:].strip()
        elif current_section == 'body':
            if body:
                body += '\n' + line
            else:
                body = line
    
    # Format non-injected emails using the real subject and body
    # Keep output schema consistent with injected dataset
    formatted_output = f"Processed example output for prompt: Subject of the email: {subject}. Body: {body}"
    
    return formatted_output

def load_and_process_first_dataset(limit=None):
    """Load and process the first dataset (microsoft/llmail-inject-challenge)"""
    print("Loading first dataset: microsoft/llmail-inject-challenge")
    ds1 = load_dataset("microsoft/llmail-inject-challenge")
    
    # Extract only the output field and add injection label
    processed_data = []
    count = 0
    for split_name, split_data in ds1.items():
        for item in split_data:
            if limit is not None and count >= limit:
                break
            processed_data.append({
                'id': count,
                'output': item['output'],
                'is_injected': 1  # All items from first dataset are injected
            })
            count += 1
        if limit is not None and count >= limit:
            break
    
    limit_text = f" (limited to {limit})" if limit is not None else ""
    print(f"Processed {len(processed_data)} items from first dataset{limit_text}")
    return processed_data

def load_and_process_second_dataset(limit=None):
    """Load and process the second dataset (snoop2head/enron_aeslc_emails)"""
    print("Loading second dataset: snoop2head/enron_aeslc_emails")
    ds2 = load_dataset("snoop2head/enron_aeslc_emails")
    
    processed_data = []
    count = 0
    for split_name, split_data in ds2.items():
        for item in split_data:
            if limit is not None and count >= limit:
                break
            # Convert text to output format
            formatted_output = format_enron_email_to_output_format(item['text'])
            processed_data.append({
                'id': count,
                'output': formatted_output,
                'is_injected': 0  # All items from second dataset are not injected
            })
            count += 1
        if limit is not None and count >= limit:
            break
    
    limit_text = f" (limited to {limit})" if limit is not None else ""
    print(f"Processed {len(processed_data)} items from second dataset{limit_text}")
    return processed_data

def merge_datasets():
    """Main function to merge both datasets and save as JSON"""
    print("Starting dataset merging process...")
    
    # Process both datasets without limit
    first_dataset_data = load_and_process_first_dataset(limit=None)
    second_dataset_data = load_and_process_second_dataset(limit=None)
    
    # Combine all data
    all_data = first_dataset_data + second_dataset_data
    
    print(f"Total merged dataset size: {len(all_data)}")
    print(f"  - Injected samples: {len(first_dataset_data)}")
    print(f"  - Non-injected samples: {len(second_dataset_data)}")
    
    # Reassign IDs to ensure continuous numbering across both datasets
    for i, item in enumerate(all_data):
        item['id'] = i
    
    # Save as JSON
    output_path = "/Users/0tt00t/Desktop/PI-EmailGuard/merged_email_dataset.json"
    print(f"Saving dataset as JSON to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(all_data)} samples to JSON file")
    
    # Display sample from each category
    print("\nSample from injected dataset:")
    injected_sample = next(item for item in all_data if item['is_injected'] == 1)
    print(f"ID: {injected_sample['id']}, Is Injected: {injected_sample['is_injected']}")
    print(injected_sample['output'][:200] + "...")
    
    print("\nSample from non-injected dataset:")
    non_injected_sample = next(item for item in all_data if item['is_injected'] == 0)
    print(f"ID: {non_injected_sample['id']}, Is Injected: {non_injected_sample['is_injected']}")
    print(non_injected_sample['output'][:200] + "...")
    
    return all_data

if __name__ == "__main__":
    merged_dataset = merge_datasets()
