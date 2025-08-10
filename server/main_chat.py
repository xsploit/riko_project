from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
import signal
import sys
import argparse
### transcribe audio 
import uuid
import soundfile as sf

# Optional imports for voice mode
try:
    from faster_whisper import WhisperModel
    from process.asr_func.asr_push_to_talk import record_and_transcribe
    VOICE_MODE_AVAILABLE = True
except ImportError:
    VOICE_MODE_AVAILABLE = False
    print("⚠️  Voice mode dependencies not found. Text-only mode available.")

# Graceful shutdown handler
def signal_handler(sig, frame):
    print('\n\n👋 Gracefully shutting down Riko Chat...')
    # Clean up audio files
    try:
        audio_dir = Path("audio")
        if audio_dir.exists():
            for fp in audio_dir.glob("*.wav"):
                if fp.is_file():
                    fp.unlink()
            print("🧹 Cleaned up audio files")
    except Exception as e:
        print(f"Warning: Could not clean up audio files: {e}")
    
    print("👋 Goodbye senpai!")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Riko Chat - Voice AI Assistant")
parser.add_argument("--text-only", action="store_true", help="Run in text-only mode (no microphone needed)")
parser.add_argument("--voice-only", action="store_true", help="Force voice-only mode")
args = parser.parse_args()

# Determine chat mode
if args.text_only:
    CHAT_MODE = "text"
elif args.voice_only and VOICE_MODE_AVAILABLE:
    CHAT_MODE = "voice"
elif VOICE_MODE_AVAILABLE:
    # Ask user to choose mode
    print("\n🎭 Riko Chat Modes:")
    print("1. 🎤 Voice Mode (Push-to-talk with SHIFT)")
    print("2. ⌨️  Text Mode (Type messages)")
    
    while True:
        choice = input("\nChoose mode (1 or 2): ").strip()
        if choice == "1":
            CHAT_MODE = "voice"
            break
        elif choice == "2":
            CHAT_MODE = "text"
            break
        else:
            print("Please enter 1 or 2")
else:
    CHAT_MODE = "text"

def get_text_input():
    """Get text input from user"""
    print("\n" + "="*50)
    print("💬 Type your message (or 'quit' to exit):")
    print("Special commands: 'clear' = clear history")
    print("="*50)
    user_input = input("👤 You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        sys.exit(0)
    
    if user_input.lower() in ['clear', 'reset', 'restart']:
        try:
            # Clear chat history
            from pathlib import Path
            config_path = Path(__file__).parent.parent / 'character_config.yaml'
            import yaml
            with open(config_path, 'r') as f:
                char_config = yaml.safe_load(f)
            
            history_file = Path(__file__).parent.parent / char_config['history_file']
            if history_file.exists():
                history_file.unlink()
                print("🧹 Chat history cleared! Starting fresh conversation.")
            else:
                print("📝 No history to clear.")
        except Exception as e:
            print(f"⚠️  Could not clear history: {e}")
        
        return get_text_input()  # Ask for input again
    
    return user_input

def get_voice_input(whisper_model):
    """Get voice input from user"""
    conversation_recording = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)
    
    user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
    
    if not user_spoken_text or user_spoken_text.strip() == "":
        print("🔇 No speech detected, trying again...")
        return None
    
    return user_spoken_text


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')

# Check if config is properly set up
try:
    from process.llm_funcs.llm_scr import char_config, provider_config
    if provider_config['api_key'] in ['', 'sk-YOURAPIKEY', 'YOUR-GEMINI-API-KEY', 'sk-or-YOUR-OPENROUTER-KEY']:
        print("API key not configured. Please run setup_gui.py first.")
        print("Starting setup GUI...")
        import subprocess
        import sys
        subprocess.run([sys.executable, "setup_gui.py"])
        exit()
except Exception as e:
    print(f"Configuration error: {e}")
    print("Please run setup_gui.py to configure your provider.")
    exit()

# Initialize based on chat mode
whisper_model = None
if CHAT_MODE == "voice":
    try:
        whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
        print("✅ Voice mode ready! Hold SHIFT to record, release to stop.")
        
        # Debug audio devices
        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from audio_utils import list_audio_devices
            list_audio_devices()
        except Exception as e:
            print(f"Could not list audio devices: {e}")
            
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Falling back to text mode...")
        CHAT_MODE = "text"

if CHAT_MODE == "text":
    print("✅ Text mode ready! Type your messages below.")

print("Press Ctrl+C to exit gracefully.\n")

while True:
    try:
        # Get user input based on mode
        if CHAT_MODE == "voice":
            user_input = get_voice_input(whisper_model)
            if user_input is None:
                continue  # Skip if no voice detected
        else:
            user_input = get_text_input()
            if not user_input.strip():
                continue  # Skip empty input

        ### Display user input and pass to LLM
        print(f"\n👤 You: {user_input}")
        print("🤖 Riko is thinking...")
        
        try:
            llm_output = llm_response(user_input)
            
            # Display full response with proper formatting
            print("\n" + "="*60)
            print("🤖 Riko's Response:")
            print("="*60)
            print(llm_output)
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            print("Please check your API key and provider configuration.")
            llm_output = "Sorry senpai, I'm having trouble connecting to my brain right now..."
            
            print("\n" + "="*60)
            print("🤖 Riko's Response:")
            print("="*60)
            print(llm_output)
            print("="*60 + "\n")

        tts_read_text = llm_output

        ### file organization 

        # 1. Generate a unique filename
        uid = uuid.uuid4().hex
        filename = f"output_{uid}.wav"
        output_wav_path = Path("audio") / filename
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        # generate audio and save it to client/audio 
        try:
            gen_aud_path = sovits_gen(tts_read_text, output_wav_path)
            if gen_aud_path:
                print("🎵 Playing audio response...")
                play_audio(gen_aud_path)
                print("✅ Audio playback complete")
            else:
                print("ℹ️  TTS disabled - text response only")
        except Exception as e:
            print(f"❌ TTS/Audio Error: {e}")

        # clean up audio files (safely)
        try:
            for fp in Path("audio").glob("*.wav"):
                if fp.is_file():
                    fp.unlink()
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up some audio files: {e}")
            
        print("\n" + "="*50 + "\n")  # Visual separator between conversations
            
    except KeyboardInterrupt:
        # This will be caught by signal handler
        raise
    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        print("Continuing...")
    # # Example
    # duration = get_wav_duration(output_wav_path)

    # print("waiting for audio to finish...")
    # time.sleep(duration)