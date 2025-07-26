import pandas as pd
import json
from typing import List, Dict

def convert_csv_to_qwen_format(csv_file_path: str, output_file_path: str = 'qwen_training_data.jsonl'):
    """
    Convert CSV conversation data to Qwen-compatible training format
    """
    
    # Load the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Sort by conversation ID and ensure proper order
    df = df.sort_values(['Conversation ID', 'Message'])
    
    training_data = []
    
    # Process each conversation
    for conv_id in df['Conversation ID'].unique():
        conv_data = df[df['Conversation ID'] == conv_id]
        
        messages = []
        
        # Add system message to maintain Riko's personality
        system_message = {
            "role": "system",
            "content": "You are Riko, a tsundere character who acts tough and dismissive but cares deep down. You often deny your feelings and get flustered easily."
        }
        messages.append(system_message)
        
        # Convert each message in the conversation
        for _, row in conv_data.iterrows():
            if row['Speaker'] == 'Rayen':
                role = "user"
            elif row['Speaker'] == 'Riko':
                role = "assistant"
            else:
                continue  # Skip unknown speakers
            
            message = {
                "role": role,
                "content": row['Message'].strip()
            }
            messages.append(message)
        
        # Create training example
        training_example = {
            "messages": messages
        }
        
        training_data.append(training_example)
    
    # Save as JSONL format (one JSON object per line)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for example in training_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(training_data)} conversations to {output_file_path}")
    return training_data

def convert_to_chatml_format(csv_file_path: str, output_file_path: str = 'qwen_chatml_data.txt'):
    """
    Convert CSV to ChatML format specifically for Qwen
    """
    
    df = pd.read_csv(csv_file_path)
    df = df.sort_values(['Conversation ID', 'Message'])
    
    chatml_data = []
    
    for conv_id in df['Conversation ID'].unique():
        conv_data = df[df['Conversation ID'] == conv_id]
        
        # Start conversation with system prompt
        conversation = "<|im_start|>system\nYou are Riko, a tsundere character who acts tough and dismissive but cares deep down. You often deny your feelings and get flustered easily.<|im_end|>\n"
        
        for _, row in conv_data.iterrows():
            if row['Speaker'] == 'Rayen':
                conversation += f"<|im_start|>user\n{row['Message'].strip()}<|im_end|>\n"
            elif row['Speaker'] == 'Riko':
                conversation += f"<|im_start|>assistant\n{row['Message'].strip()}<|im_end|>\n"
        
        # Add EOS token
        conversation += "<|endoftext|>\n\n"
        chatml_data.append(conversation)
    
    # Save ChatML format
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(chatml_data)
    
    print(f"Converted {len(chatml_data)} conversations to ChatML format: {output_file_path}")
    return chatml_data

# Example usage for Unsloth/SFT training
def prepare_for_unsloth(csv_file_path: str, output_file_path: str = 'unsloth_data.json'):
    """
    Prepare data specifically for Unsloth SFT format
    """
    
    df = pd.read_csv(csv_file_path)
    df = df.sort_values(['Conversation ID', 'Message'])
    
    unsloth_data = []
    
    for conv_id in df['Conversation ID'].unique():
        conv_data = df[df['Conversation ID'] == conv_id]
        
        # For each user-assistant pair
        user_messages = conv_data[conv_data['Speaker'] == 'Rayen']
        assistant_messages = conv_data[conv_data['Speaker'] == 'Riko']
        
        # Create instruction-response pairs
        for i, (_, user_row) in enumerate(user_messages.iterrows()):
            # Find corresponding assistant response
            assistant_responses = assistant_messages[assistant_messages.index > user_row.name]
            
            if not assistant_responses.empty:
                assistant_row = assistant_responses.iloc[0]
                
                training_example = {
                    "instruction": user_row['Message'].strip(),
                    "input": "",  # Empty for single-turn format
                    "output": assistant_row['Message'].strip()
                }
                
                unsloth_data.append(training_example)
    
    # Save as JSON
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(unsloth_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(unsloth_data)} training examples for Unsloth: {output_file_path}")
    return unsloth_data

# Main execution
if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual CSV file path
    csv_file = 'Riko.csv'
    
    print("Converting data for Qwen fine-tuning...")
    
    # Method 1: Standard JSONL format
    convert_csv_to_qwen_format(csv_file, 'qwen_training_data.jsonl')
    
    # Method 2: ChatML format
    convert_to_chatml_format(csv_file, 'qwen_chatml_data.txt')
    
    # Method 3: Unsloth format (if using Unsloth)
    prepare_for_unsloth(csv_file, 'unsloth_data.json')
    
    print("\nAll formats created successfully!")
    print("Files generated:")
    print("- qwen_training_data.jsonl (for standard fine-tuning)")
    print("- qwen_chatml_data.txt (ChatML format)")
    print("- unsloth_data.json (for Unsloth SFT)")