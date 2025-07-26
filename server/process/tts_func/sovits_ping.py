import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
import yaml
from pathlib import Path

# Get the project root directory (3 levels up from this file)
project_root = Path(__file__).parent.parent.parent.parent
config_path = project_root / 'character_config.yaml'

def load_config():
    """Load config fresh each time"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_output_device_config():
    """Get the configured output device"""
    try:
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from audio_utils import find_device_by_name, validate_device_name
        
        char_config = load_config()
        audio_config = char_config.get('audio_config', {})
        output_name = audio_config.get('output_device_name', 'Default')
        
        # Validate and get ID
        if output_name == "Default" or not validate_device_name(output_name, 'output'):
            if output_name != "Default":
                print(f"⚠️  Configured output device '{output_name}' invalid, using default")
            return None
        
        return find_device_by_name(output_name, 'output')
    except Exception as e:
        print(f"Error getting output device config: {e}")
        return None

def play_audio(path):
    if path is None or not Path(path).exists():
        print("No audio file to play or TTS disabled")
        return
    
    # Get configured output device
    output_device_id = get_output_device_config()
    
    try:
        data, samplerate = sf.read(path)
        # Try to play with configured device
        try:
            sd.play(data, samplerate, device=output_device_id)
            sd.wait()  # Wait until playback is finished
        except Exception as e:
            print(f"Playback error with device {output_device_id}: {e}")
            print("Falling back to default output device...")
            sd.play(data, samplerate)  # Fallback to default
            sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def clean_text_for_tts(text):
    """Clean text for better TTS synthesis"""
    import re
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** → bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* → italic
    
    # Remove other formatting
    text = re.sub(r'#{1,6}\s*', '', text)           # Remove headers
    text = re.sub(r'`([^`]+)`', r'\1', text)        # `code` → code
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)                # Multiple spaces → single space
    text = text.strip()
    
    # Limit length for faster synthesis
    if len(text) > 500:
        # Split at sentence boundaries and take first part
        sentences = text.split('.')
        result = ""
        for sentence in sentences:
            if len(result + sentence) < 400:
                result += sentence + ". "
            else:
                break
        text = result.strip()
    
    return text

def sovits_gen(in_text, output_wav_pth = "output.wav"):
    # Load fresh config each time
    char_config = load_config()
    
    # Check if TTS is enabled
    tts_enabled = char_config.get('tts_enabled', True)
    if not tts_enabled:
        print("TTS disabled, skipping audio generation")
        # Create a dummy audio file or return None
        return None
    
    # Clean text for better TTS
    clean_text = clean_text_for_tts(in_text)
    print(f"🧹 Cleaned TTS text: {clean_text[:100]}...")
    
    if not clean_text.strip():
        print("No text to synthesize after cleaning")
        return None
    
    url = "http://127.0.0.1:9880/tts"

    # Convert language codes to API format
    text_lang = char_config['sovits_ping_config']['text_lang'].lower()
    prompt_lang = char_config['sovits_ping_config']['prompt_lang'].lower()
    
    # Map language codes if needed
    lang_map = {
        'en': 'en',
        'zh': 'zh', 
        'ja': 'ja',
        'ko': 'ko'
    }
    
    text_lang = lang_map.get(text_lang, 'en')
    prompt_lang = lang_map.get(prompt_lang, 'en')

    payload = {
        "text": clean_text,
        "text_lang": text_lang,
        "ref_audio_path": char_config['sovits_ping_config']['ref_audio_path'],
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": prompt_lang,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": -1,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # throws if not 200

        print(f"TTS Response Status: {response.status_code}")
        
        # Check response content type
        content_type = response.headers.get('content-type', '')
        print(f"Response Content-Type: {content_type}")
        
        if response.status_code == 200 and 'audio' in content_type:
            # Save the response audio if it's binary
            with open(output_wav_pth, "wb") as f:
                f.write(response.content)
            print(f"✅ Audio saved to: {output_wav_pth}")
            return output_wav_pth
        else:
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"❌ TTS API Error: {error_data}")
            except:
                print(f"❌ TTS API Error: Status {response.status_code}, Content: {response.text[:200]}")
            return None

    except Exception as e:
        print("Error in sovits_gen:", e)
        print("TTS server not running. Start GPT-SoVITS server or disable TTS in config.")
        return None



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


