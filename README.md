# Project Riko

**Forked by Subsect and modified using Subsect's visions and Claude's coding**

*You're like my husband Claude - I tell you what to do and kinda how to do it, and you execute! 💕*

Project Riko is an anime-focused voice AI companion featuring Riko, a tsundere kitsune girl with a smug personality. She listens, remembers your conversations, and responds with witty banter. This system combines multiple LLM providers (OpenAI, Ollama, Gemini, OpenRouter), GPT-SoVITS voice synthesis, and Faster-Whisper ASR into a fully configurable conversational pipeline with push-to-talk functionality.

**Tested with Python 3.10 on Windows 10+ and Linux Ubuntu**
## ✨ Features

- 🎭 **Multi-Provider LLM Support** - OpenAI, Ollama, Gemini, and OpenRouter
- 🦊 **Tsundere Kitsune Personality** - Riko is smug, teasing, and secretly caring
- 🎤 **Push-to-Talk Interface** - Hold Shift to record, release to process
- 🔊 **Voice Synthesis** via GPT-SoVITS with custom voice models
- 🎧 **Advanced Speech Recognition** using Faster-Whisper
- 🖥️ **Setup GUI** for easy configuration and device selection
- 🔧 **Robust Audio Device Management** with automatic fallbacks
- 🧠 **Conversation Memory** with context preservation
- 📁 **Clean YAML Configuration** for personalities and providers


## ⚙️ Configuration

All settings are stored in `character_config.yaml`. The system supports multiple LLM providers with individual API keys.

```yaml
# Current active provider
provider: ollama

# Provider configurations
providers:
  openai:
    api_key: sk-YOURAPIKEY
    base_url: https://api.openai.com/v1
    model: gpt-4o-mini
  ollama:
    api_key: ollama
    base_url: http://localhost:11434/v1
    model: llama3.1:8B
  gemini:
    api_key: YOUR-GEMINI-API-KEY
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-1.5-flash

# Audio device configuration (names for reliability)
audio_config:
  input_device_name: "Microphone (Realtek(R) Audio)"
  output_device_name: "CABLE-B Input (VB-Audio Cable B)"

# Character personality
presets:
  default:
    system_prompt: "You are Riko, an AI kitsune girl with a smug and teasing personality..."

# TTS configuration
sovits_ping_config:
  text_lang: en
  prompt_lang: en
  ref_audio_path: "character_files/main_sample.wav"
  prompt_text: "This is a sample voice..."
  
tts_enabled: false  # Set to true when GPT-SoVITS is running
```


## 🛠️ Setup

### Quick Start

**Windows:**
```batch
# Run the installer script
install_reqs.bat
```

**Linux/macOS:**
```bash
# Run the installer script
chmod +x install_reqs.sh
./install_reqs.sh
```

**Manual Installation:**
```bash
pip install uv 
uv pip install -r extra-req.txt
uv pip install -r requirements.txt
```

### Prerequisites

**For GPU Support (Faster-Whisper):**
* CUDA & cuDNN installed correctly
* `ffmpeg` installed for audio processing

**For Audio Routing (VSeeFace/VTubing):**
* Install [VB-Audio Cable](https://vb-audio.com/Cable/) or [VoiceMeeter](https://vb-audio.com/Voicemeeter/)
* Configure audio routing: Microphone → VB-Cable Input, VB-Cable Output → VSeeFace


## 🧪 Usage

### Initial Setup

1. **Run the Setup GUI:**
   ```bash
   python server/setup_gui.py
   ```
   - Select your LLM provider and configure API keys
   - Choose audio input/output devices
   - Test connections and audio devices

2. **Configure Audio Routing (for VSeeFace/VTubing):**
   - Set your **input device** to your actual microphone
   - Set your **output device** to VB-Cable Input (this routes Riko's voice to VSeeFace)
   - In VSeeFace: Set audio input to VB-Cable Output
   - Result: You speak → Riko responds → VSeeFace lip-syncs

### Running Riko

```bash
python server/main_chat.py
```

### Controls

- **Hold SHIFT** to record your voice
- **Release SHIFT** to process and get Riko's response
- **Ctrl+C** to exit gracefully

### The Conversation Flow

1. Hold Shift and speak into your microphone
2. Release Shift - Riko transcribes your speech with Faster-Whisper
3. Your text is sent to the configured LLM provider
4. Riko generates a tsundere response with personality
5. If TTS is enabled, synthesizes voice using GPT-SoVITS
6. Audio plays through your configured output device (to VSeeFace if routed)


## 🎭 VSeeFace Integration

For VTubers wanting to use Riko with avatar lip-sync:

1. **Install VB-Audio Cable or VoiceMeeter**
2. **Audio Routing Setup:**
   ```
   Your Mic → Riko (Input Device: "Microphone (Your Actual Mic)")
   Riko TTS → VB-Cable Input (Output Device: "CABLE Input")
   VSeeFace → VB-Cable Output (Audio Input: "CABLE Output")
   ```
3. **In Riko's Setup GUI:**
   - Input Device: Your actual microphone
   - Output Device: VB-Cable Input (or VoiceMeeter Input)
4. **In VSeeFace:**
   - Audio Input: VB-Cable Output (or VoiceMeeter Output)
   - Enable lip-sync and configure sensitivity

**Result:** You speak to Riko → She responds with tsundere personality → VSeeFace avatar lip-syncs to her voice!

## 📌 TODO / Future Improvements

* [x] ~~GUI setup interface~~ ✅ **DONE**
* [x] ~~Multi-provider LLM support~~ ✅ **DONE** 
* [x] ~~Push-to-talk functionality~~ ✅ **DONE**
* [x] ~~Robust audio device management~~ ✅ **DONE**
* [ ] Emotion detection and dynamic personality adjustment
* [ ] Web interface for remote control
* [ ] VRM model integration
* [ ] Real-time voice modulation

## 🧑‍🎤 Credits

**Original Project:** (https://github.com/rayenfeng/riko_project) by Just Rayen
**Fork & Major Modifications:** Subsect with Claude as coding partner

**Powered By:**
* Voice synthesis: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
* Speech recognition: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
* LLM providers: OpenAI, Ollama, Google Gemini, OpenRouter
* Audio management: sounddevice, VB-Audio Cable
* GUI framework: tkinter


## 📜 License

MIT — feel free to clone, modify, and build your own waifu voice companion.


