from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
import signal
import sys
### transcribe audio 
import uuid
import soundfile as sf

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

try:
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please ensure the model is available or check your internet connection.")
    exit()

print("✅ System ready! Press Ctrl+C to exit gracefully.\n")

# Debug audio devices
try:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from audio_utils import list_audio_devices
    list_audio_devices()
except Exception as e:
    print(f"Could not list audio devices: {e}")

while True:
    try:
        conversation_recording = output_wav_path = Path("audio") / "conversation.wav"
        conversation_recording.parent.mkdir(parents=True, exist_ok=True)

        user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
        
        # Skip empty transcriptions
        if not user_spoken_text or user_spoken_text.strip() == "":
            print("🔇 No speech detected, trying again...")
            continue

        ### Display user input and pass to LLM
        print(f"\n👤 You: {user_spoken_text}")
        print("🤖 Riko is thinking...")
        
        try:
            llm_output = llm_response(user_spoken_text)
            print(f"🤖 Riko: {llm_output}")
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            print("Please check your API key and provider configuration.")
            llm_output = "Sorry senpai, I'm having trouble connecting to my brain right now..."
            print(f"🤖 Riko: {llm_output}")

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