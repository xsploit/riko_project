# OpenAI tool calling with history 
### Uses a sample function
import yaml
import gradio as gr
import json
import os
from pathlib import Path
from openai import OpenAI

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent
config_path = project_root / 'character_config.yaml'

with open(config_path, 'r') as f:
    char_config = yaml.safe_load(f)

# Get current provider configuration
current_provider = char_config['provider']
provider_config = char_config['providers'][current_provider]

# Export for validation
__all__ = ['char_config', 'provider_config', 'llm_response']

client = OpenAI(
    api_key=provider_config['api_key'],
    base_url=provider_config['base_url']
)

# Constants
HISTORY_FILE = project_root / char_config['history_file']
MODEL = provider_config['model']
SYSTEM_PROMPT =  [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": char_config['presets']['default']['system_prompt']  
                }
            ]
        }
    ]

# Load/save chat history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return SYSTEM_PROMPT

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)



def get_riko_response_no_tool(messages):

    # Convert to OpenAI-compatible format
    openai_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            openai_messages.append({
                'role': 'system',
                'content': msg['content'][0]['text']
            })
        elif msg['role'] == 'user':
            openai_messages.append({
                'role': 'user', 
                'content': msg['content'][0]['text']
            })
        elif msg['role'] == 'assistant':
            openai_messages.append({
                'role': 'assistant',
                'content': msg['content'][0]['text']
            })
    
    # Call OpenAI-compatible API
    response = client.chat.completions.create(
        model=MODEL,
        messages=openai_messages,
        temperature=1,
        top_p=1,
        max_tokens=2048,
        stream=False
    )

    return response


def llm_response(user_input):

    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })


    riko_test_response = get_riko_response_no_tool(messages)


    # just append assistant message to regular response. 
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": riko_test_response.choices[0].message.content}
    ]
    })

    save_history(messages)
    return riko_test_response.choices[0].message.content


if __name__ == "__main__":
    print('running main')