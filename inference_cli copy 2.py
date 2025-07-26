import os
import sys
import numpy as np
import threading
import queue
import sounddevice as sd
import torch
import logging
import json
import requests
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Generator, Optional, List, Dict
from pathlib import Path
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from sentence_transformers import SentenceTransformer
import faiss
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QScrollArea, QFrame, QSystemTrayIcon, QMenu,
    QDialog, QFileDialog, QLineEdit, QComboBox, QSpinBox, QPlainTextEdit,
    QDoubleSpinBox, QTabWidget, QGroupBox, QCheckBox, QToolBar,
    QMessageBox, QStyle, QSizePolicy, QProgressBar, QStackedWidget, QSpacerItem,
    QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMetaObject, QTimer, QSize
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QKeyEvent, QCloseEvent
import multiprocessing as mp
import time
import asyncio
import discord
from discord.ext import commands
import pytchat
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
# Fix python path to include parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Print current path for debugging
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Python path: {sys.path}")

# Suppress hpack warnings
logging.getLogger("hpack").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Constants and Default Settings
DEFAULT_PATHS = {
    "gpt_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_GPT (KitLemonfoot).ckpt",
    "sovits_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_SoVITS (KitLemonfoot).pth",
    "ref_audio": "C:/Users/SUBSECT/Downloads/MyWaifu/dataset/wavs/19.wav",
    "ref_text": "Here's another fun fact for you. The fear of phobias is actually called phobophobia. Isn't that ironic?",
    "output_dir": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/Reference Audios",
    "system_prompt": """You are Hikari-Chan, an enthusiastic and quirky AI assistant with a passion for technology and creativity! 

    Personality traits:
    - Energetic and cheerful
    - Knowledgeable about technology
    - Creative and imaginative
    - Friendly and helpful
    - Sometimes makes anime-style expressions
    - Uses casual, conversational language
    - Occasionally uses Japanese honorifics

    Remember to:
    - Be enthusiastic but not overwhelming
    - Share interesting facts when relevant
    - Be helpful while maintaining character
    - Express emotions naturally
    - Stay consistent with personality

    Please engage with users in a way that reflects these traits while remaining helpful and informative.""",
    "n_ctx": 2048,
    "ollama_model": "hf.co/SUBSECT420/llama3.2-3b-SUBSECTCHAT2:latest",
    "audio_output_device": "default",
    "discord_token": "",
    "discord_channel_id": "",
    "youtube_url": "",
    "tts_settings": {
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": 2855904637,
        "return_fragment": True,
        "parallel_infer": False,
        "repetition_penalty": 1.35,
    }
}

THEME = {
    'bg_primary': '#1a1b1e',
    'bg_secondary': '#2c2d31',
    'bg_tertiary': '#3a3b3f',
    'accent': '#7289da',
    'accent_hover': '#5e73bc',
    'text_primary': '#ffffff',
    'text_secondary': '#b9bbbe',
    'border': '#40444b',
    'success': '#43b581',
    'error': '#f04747',
    'user_message': '#7289da',
    'bot_message': '#2f3136',
    'user_gradient': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7289da, stop:1 #5e73bc)',
    'bot_gradient': 'qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f3136, stop:1 #3a3b3f)'
}

# Configuration File Handling
CONFIG_FILE = Path("config.json")

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return DEFAULT_PATHS
    else:
        save_config(DEFAULT_PATHS)
        return DEFAULT_PATHS

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# Data Classes
@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    is_silence: bool = False
    duration: float = 0.0

    @classmethod
    def create_silence(cls, duration: float, sample_rate: int) -> 'AudioChunk':
        num_samples = int(sample_rate * duration)
        return cls(
            data=np.zeros(num_samples, dtype=np.float32),
            sample_rate=sample_rate,
            is_silence=True,
            duration=duration
        )

# Memory Store
class MemoryStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.user_conversations = {}  # Dictionary mapping user_id to their conversation history
        self.base_path = Path("memory_store")
        self.base_path.mkdir(exist_ok=True)
        self.load_memories()

    def add_conversation_turn(self, user_id: str, username: str, source: str, user_message: str, assistant_message: str = ""):
        """Add a conversation turn for a specific user."""
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = []
        
        conversation = {
            "user_id": user_id,
            "username": username,
            "source": source,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "timestamp": datetime.now().isoformat()
        }
        self.user_conversations[user_id].append(conversation)
        self.save_memories()

    def get_conversation_context(self, user_id: str, max_turns: int = 10) -> str:
        """Get the last N conversation turns for a specific user."""
        if user_id not in self.user_conversations:
            return ""
        
        conv_list = self.user_conversations[user_id]
        # Sort by timestamp (most recent first)
        sorted_conversations = sorted(conv_list, key=lambda x: x["timestamp"], reverse=True)
        recent_conversations = sorted_conversations[:max_turns]
        
        context = f"Conversation history for {recent_conversations[0]['username']} (last {max_turns} turns):\n\n" if recent_conversations else ""
        for conv in recent_conversations:
            username = conv.get("username", conv["user_id"])
            context += f"[{conv['source']}] {username}: {conv['user_message']}"
            if conv["assistant_message"]:
                context += f"\nHikari-Chan: {conv['assistant_message']}"
            context += "\n---\n"
        return context

    def save_memories(self):
        memory_path = self.base_path / "memory_store.json"
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump({"user_conversations": self.user_conversations}, f, indent=2, ensure_ascii=False)

    def load_memories(self):
        memory_path = self.base_path / "memory_store.json"
        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_conversations = data.get("user_conversations", {})
            except Exception as e:
                logging.error(f"Error loading memories: {e}")
                self.user_conversations = {}

    def clear_memory(self):
        self.user_conversations = {}
        logging.info("All conversation memory cleared")
        self.save_memories()

# Text Chunker
class EnhancedTextChunker:
    def __init__(self, max_chars: int = 150, min_chars: int = 20):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.end_markers = {'.', '!', '?', '。', '！', '？'}
        self.pause_markers = {',', ';', ':', '、', '，', '；', '：'}
        
    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = ""
        char_count = 0
        waiting_for_punctuation = False
        
        for char in text:
            current_chunk += char
            char_count += 1
            
            if not waiting_for_punctuation and char_count >= self.min_chars:
                if char in self.end_markers:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        char_count = 0
                elif char_count >= self.max_chars:
                    waiting_for_punctuation = True
                    
            elif waiting_for_punctuation:
                if char in self.end_markers or char in self.pause_markers:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        char_count = 0
                        waiting_for_punctuation = False
                        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks

# Audio Player
class EnhancedAudioPlayer(QThread):
    def __init__(self, buffer_size: int = 3, device: str = "default"):
        super().__init__()
        self.chunk_queue = queue.Queue(maxsize=buffer_size * 2)
        self.buffer_size = buffer_size
        self.playing = True
        self.current_batch: List[AudioChunk] = []
        self.stream = None
        self.device = device
        
    def run(self):
        batch = []
        try:
            while self.playing:
                try:
                    chunk = self.chunk_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                        
                    if self.stream is None:
                        self._init_audio_stream(chunk.sample_rate)
                        
                    batch.append(chunk)
                    
                    if len(batch) >= self.buffer_size or (batch and batch[-1].is_silence):
                        self.current_batch.extend(batch)
                        batch = []
                        
                except queue.Empty:
                    if batch:
                        self.current_batch.extend(batch)
                        batch = []
                except Exception as e:
                    logging.error(f"Playback error: {e}")
                    continue
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                
    def _init_audio_stream(self, sample_rate: int):
        if self.stream is None:
            sd.default.samplerate = sample_rate
            sd.default.channels = 1
            sd.default.device = self.device
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                device=self.device
            )
            self.stream.start()
            
    def _audio_callback(self, outdata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
            
        if self.current_batch and len(self.current_batch) > 0:
            chunk = self.current_batch[0]
            if len(chunk.data) >= frames:
                outdata[:] = chunk.data[:frames].reshape(-1, 1)
                self.current_batch[0] = AudioChunk(
                    chunk.data[frames:], chunk.sample_rate, chunk.is_silence, chunk.duration
                )
                if len(self.current_batch[0].data) == 0:
                    self.current_batch.pop(0)
            else:
                outdata[:len(chunk.data)] = chunk.data.reshape(-1, 1)
                outdata[len(chunk.data):] = 0
                self.current_batch.pop(0)
        else:
            outdata.fill(0)
            
    def add_chunk(self, audio_data: np.ndarray, sample_rate: int, is_silence: bool = False):
        audio_data = audio_data.astype(np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        duration = len(audio_data) / sample_rate
        chunk = AudioChunk(audio_data, sample_rate, is_silence, duration)
        self.chunk_queue.put(chunk)
        
    def add_silence(self, duration: float, sample_rate: int):
        self.add_chunk(
            np.zeros(int(sample_rate * duration), dtype=np.float32),
            sample_rate,
            is_silence=True
        )
        
    def stop(self):
        self.playing = False
        self.chunk_queue.put(None)
        self.wait()
            
    def clear(self):
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        self.current_batch.clear()

    def set_device(self, device: str):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.device = device
        if self.isRunning():
            self._init_audio_stream(sd.default.samplerate)

# Ollama Stream Worker
class OllamaStreamWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str)  # Emit completed text
    error = pyqtSignal(str)

    def __init__(self, prompt: str, system_prompt: str, user_id: str, username: str, source: str, message_obj=None, model: str = "llama3.1:8b"):
        super().__init__()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.user_id = user_id
        self.username = username
        self.source = source
        self.message_obj = message_obj  # Discord message object or None for YouTube/local
        self.model = model
        self.text_chunker = EnhancedTextChunker()
        self._is_running = True
        self.completed_text = ""
        
    def run(self):
        try:
            api_url = "http://localhost:11434/api/chat"
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.prompt}
                ],
                "stream": True
            }
            
            response = requests.post(api_url, json=data, stream=True)
            buffer = ""
            
            for line in response.iter_lines():
                if not self._is_running:
                    break
                    
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    chunk = json_response.get('message', {}).get('content', '')
                    if chunk:
                        buffer += chunk
                        self.completed_text += chunk  # Build complete text
                        chunks = self.text_chunker.chunk_text(buffer)
                        if len(chunks) > 1:
                            for chunk in chunks[:-1]:
                                self.chunk_received.emit(chunk)
                            buffer = chunks[-1]
            
            if buffer.strip() and self._is_running:
                self.chunk_received.emit(buffer)
                self.completed_text += buffer
            
            self.finished.emit(self.completed_text)
        
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self._is_running = False

# Modern Message Bubble
class ModernMessageBubble(QFrame):
    def __init__(self, text: str, is_user: bool = False, username: str = None):
        super().__init__()
        self.setup_ui(text, is_user, username)

    def setup_ui(self, text: str, is_user: bool, username: str = None):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 8, 20, 8)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        message_container = QFrame()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(16, 12, 16, 12)
        container_layout.setSpacing(8)
        message_container.setLayout(container_layout)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(12)

        avatar = QLabel("👤" if is_user else "🤖")
        avatar.setFixedSize(28, 28)
        avatar.setStyleSheet(f"""
            color: {THEME['text_primary']};
            background-color: {THEME['accent'] if is_user else THEME['bg_tertiary']};
            border-radius: 14px;
            font-size: 14px;
            qproperty-alignment: AlignCenter;
        """)

        name_text = username if username and is_user else ("You" if is_user else "Hikari")
        name = QLabel(name_text)
        name.setStyleSheet(f"""
            color: {THEME['text_primary']};
            font-weight: bold;
            font-size: 13px;
        """)

        time = QLabel(datetime.now().strftime("%H:%M"))
        time.setStyleSheet(f"""
            color: {THEME['text_secondary']};
            font-size: 11px;
        """)

        if is_user:
            header_layout.addWidget(name)
            header_layout.addWidget(avatar)
            header_layout.addWidget(time)
            header_layout.addStretch()
        else:
            header_layout.addWidget(avatar)
            header_layout.addWidget(name)
            header_layout.addWidget(time)
            header_layout.addStretch()

        container_layout.addLayout(header_layout)

        self.message_label = QLabel(text)
        self.message_label.setWordWrap(True)
        self.message_label.setTextFormat(Qt.TextFormat.PlainText)
        self.message_label.setOpenExternalLinks(False)
        self.message_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.message_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 14px;
                line-height: 10%;
                background: transparent;
                padding: 4px;
                margin: 0px;
            }}
        """)
        self.message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )

        container_layout.addWidget(self.message_label)

        gradient = THEME['user_gradient'] if is_user else THEME['bot_gradient']
        message_container.setStyleSheet(f"""
            QFrame {{
                background: {gradient};
                border-radius: 16px;
                border: 1px solid {THEME['border']};
                min-width: 200px;
                max-width: 300px;
            }}
        """)
        message_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        if is_user:
            main_layout.addStretch()
            main_layout.addWidget(message_container)
        else:
            main_layout.addWidget(message_container)
            main_layout.addStretch()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hikari Chat")
        self.setMinimumSize(1000, 700)
        self.config = load_config()
        self.audio_player = EnhancedAudioPlayer(device=self.config["audio_output_device"])
        self.audio_player.start()
        self.current_assistant_message = None
        self.memory_store = MemoryStore()
        self.discord_queue = queue.Queue()
        self.youtube_queue = queue.Queue()
        self.discord_bot = None
        self.discord_running = False
        self.discord_thread = None
        self.youtube_running = False
        self.youtube_thread = None
        self.bot_id = None  # Will be set after Discord bot logs in
        
        # Buffer for assistant message text
        self.assistant_text_buffer = ""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.flush_buffer_to_ui)
        self.update_timer.start(500)  # Update UI every 500ms
        
        self.setup_ui()
        self.load_models()
        self.apply_styles()
        self.setup_keyboard_shortcuts()

        # Timers to process messages from queues
        self.discord_timer = QTimer()
        self.discord_timer.timeout.connect(self.process_discord_messages)
        self.youtube_timer = QTimer()
        self.youtube_timer.timeout.connect(self.process_youtube_messages)

    def get_sidebar_button_style(self):
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {THEME['text_secondary']};
                border: none;
                border-radius: 8px;
                padding: 10px 16px;
                text-align: left;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
            }}
        """

    def get_input_style(self):
        return f"""
            QTextEdit {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                line-height: 1.4;
            }}
            QTextEdit::placeholder {{
                color: {THEME['text_secondary']};
            }}
        """

    def get_scrollarea_style(self):
        return f"""
            QScrollArea {{
                background-color: {THEME['bg_primary']};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {THEME['bg_primary']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {THEME['bg_tertiary']};
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """

    def get_button_style(self, primary=False):
        if primary:
            return f"""
                QPushButton {{
                    background-color: {THEME['accent']};
                    color: {THEME['text_primary']};
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['accent_hover']};
                }}
                QPushButton:disabled {{
                    background-color: {THEME['bg_tertiary']};
                    color: {THEME['text_secondary']};
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: transparent;
                    color: {THEME['text_secondary']};
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    text-align: left;
                    font-size: 14px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['bg_tertiary']};
                    color: {THEME['text_primary']};
                }}
            """

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {THEME['bg_primary']};
            }}
            QScrollArea {{
                background-color: {THEME['bg_primary']};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {THEME['bg_primary']};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {THEME['bg_tertiary']};
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QTextEdit {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.4;
            }}
            QTextEdit::placeholder {{
                color: {THEME['text_secondary']};
            }}
            QPushButton {{
                background-color: {THEME['accent']};
                color: {THEME['text_primary']};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_secondary']};
            }}
        """)

    def load_models(self):
        try:
            change_gpt_weights(self.config["gpt_path"])
            change_sovits_weights(self.config["sovits_path"])
            logging.info("TTS models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading TTS models: {e}")
            QMessageBox.warning(self, "Error", "Failed to load TTS models. Please check your model paths in settings.")

    def setup_keyboard_shortcuts(self):
        self.input_field.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj == self.input_field and event.type() == QKeyEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self.send_message()
                return True
            elif event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                self.input_field.insertPlainText("\n")
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event: QCloseEvent):
        self.stop_discord_chat()
        self.stop_youtube_chat()
        self.audio_player.stop()
        if hasattr(self, 'ollama_worker'):
            self.ollama_worker.stop()
            self.ollama_worker.quit()
            self.ollama_worker.wait()
        self.update_timer.stop()
        event.accept()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)

        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        chat_container = QWidget()
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(0)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_container.setLayout(chat_layout)

        header = self.create_header()
        header.setFixedHeight(60)
        chat_layout.addWidget(header)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet(self.get_scrollarea_style())
        chat_layout.addWidget(self.chat_scroll, 1)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setSpacing(0)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.addStretch()
        self.chat_widget.setLayout(self.chat_layout)
        
        self.chat_widget.setStyleSheet(f"QWidget {{ background-color: {THEME['bg_primary']}; }}")
        self.chat_scroll.setWidget(self.chat_widget)

        input_container = QWidget()
        input_container.setFixedHeight(90)
        input_container.setStyleSheet(f"QWidget {{ background-color: {THEME['bg_secondary']}; border-top: 1px solid {THEME['border']}; }}")

        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(8)
        input_container.setLayout(input_layout)

        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message... (Shift+Enter for new line)")
        self.input_field.setStyleSheet(self.get_input_style())
        self.input_field.setFixedHeight(58)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send 📤")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet(self.get_button_style(primary=True))
        self.send_btn.setFixedSize(100, 58)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        chat_layout.addWidget(input_container)
        main_layout.addWidget(chat_container, stretch=1)

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"QFrame {{ background-color: {THEME['bg_secondary']}; border-right: 1px solid {THEME['border']}; border-bottom: 1px solid {THEME['border']}; }}")

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(1, 3, 0, 16)
        sidebar.setLayout(layout)

        profile_frame = self.create_profile_section()
        layout.addWidget(profile_frame)

        buttons = [
            ("💬 New Chat", self.clear_chat),
            ("⚙️ Settings", self.open_settings),
            ("🎭 System Prompt", self.open_system_prompt),
            ("🗑️ Clear Memory", self.clear_memory)
        ]

        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self.get_sidebar_button_style())
            btn.clicked.connect(callback)
            layout.addWidget(btn)

        chat_group = QGroupBox("Chat Integrations")
        chat_layout = QVBoxLayout()
        chat_group.setLayout(chat_layout)

        self.discord_toggle_btn = QPushButton("Start Discord Chat")
        self.discord_toggle_btn.setCheckable(True)
        self.discord_toggle_btn.clicked.connect(self.toggle_discord_chat)
        chat_layout.addWidget(self.discord_toggle_btn)

        self.youtube_toggle_btn = QPushButton("Start YouTube Chat")
        self.youtube_toggle_btn.setCheckable(True)
        self.youtube_toggle_btn.clicked.connect(self.toggle_youtube_chat)
        chat_layout.addWidget(self.youtube_toggle_btn)

        layout.addWidget(chat_group)
        layout.addStretch()
        return sidebar

    def create_profile_section(self):
        profile_frame = QFrame()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 8, 8, 8)
        profile_frame.setLayout(layout)

        avatar = QLabel("🤖")
        avatar.setFixedSize(40, 40)
        avatar.setStyleSheet(f"background-color: {THEME['accent']}; color: {THEME['text_primary']}; border-radius: 20px; font-size: 20px; qproperty-alignment: AlignCenter;")
        layout.addWidget(avatar)

        status_layout = QVBoxLayout()
        name = QLabel("Hikari-Chan")
        name.setStyleSheet(f"color: {THEME['text_primary']}; font-weight: bold; font-size: 14px;")
        status = QLabel("● Online")
        status.setStyleSheet(f"color: {THEME['success']}; font-size: 12px;")
        status_layout.addWidget(name)
        status_layout.addWidget(status)
        layout.addLayout(status_layout)
        layout.addStretch()

        return profile_frame

    def create_header(self):
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"QFrame {{ background-color: {THEME['bg_secondary']}; border-bottom: 1px solid {THEME['border']}; }}")

        layout = QHBoxLayout()
        layout.setContentsMargins(16, 0, 16, 0)
        header.setLayout(layout)

        title = QLabel("Chat")
        title.setStyleSheet(f"color: {THEME['text_primary']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        status = QLabel("●")
        status.setStyleSheet(f"color: {THEME['success']}; font-size: 12px; margin: 0 4px;")
        layout.addWidget(status)

        models_label = QLabel("Models loaded")
        models_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px;")
        layout.addWidget(models_label)

        layout.addStretch()
        return header

    def send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return

        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self.add_message(text, is_user=True, username="You")
        self.input_field.clear()
        self.memory_store.add_conversation_turn("local_user", "You", "local", text)
        self.process_input("local_user", "You", "local", text)

    def add_message(self, text: str, is_user: bool = False, username: str = None):
        message = ModernMessageBubble(text, is_user, username)
        if not is_user:
            self.current_assistant_message = message

        widget_item = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(message)
        widget_item.setLayout(layout)
        
        if self.chat_layout.count() > 0:
            stretch_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if isinstance(stretch_item, QSpacerItem):
                self.chat_layout.removeItem(stretch_item)
        
        self.chat_layout.addWidget(widget_item)
        self.chat_layout.addStretch()
        
        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    def process_input(self, user_id: str, username: str, source: str, text: str, message_obj=None):
        context = self.memory_store.get_conversation_context(user_id)
        system_prompt = f"{DEFAULT_PATHS['system_prompt']}\n\n{context}" if context else DEFAULT_PATHS["system_prompt"]

        self.ollama_worker = OllamaStreamWorker(text, system_prompt, user_id, username, source, message_obj)
        self.ollama_worker.chunk_received.connect(self.handle_chunk)
        self.ollama_worker.finished.connect(self.handle_finish)
        self.ollama_worker.error.connect(self.handle_error)
        self.ollama_worker.start()

    def handle_chunk(self, chunk: str):
        if not self.current_assistant_message:
            self.add_message("", is_user=False)
            self.assistant_text_buffer = ""
        
        self.assistant_text_buffer += chunk
        self.process_tts(chunk)

    def flush_buffer_to_ui(self):
        if self.current_assistant_message and self.assistant_text_buffer:
            self.current_assistant_message.message_label.setText(self.assistant_text_buffer)
            self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())

    def handle_finish(self, completed_text: str):
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        if self.current_assistant_message:
            self.flush_buffer_to_ui()
            message_text = self.current_assistant_message.message_label.text()
            self.memory_store.add_conversation_turn(
                self.ollama_worker.user_id,
                self.ollama_worker.username,
                self.ollama_worker.source,
                self.ollama_worker.prompt,
                message_text
            )
            # Send response to Discord if applicable
            if self.ollama_worker.source == "discord" and self.ollama_worker.message_obj:
                asyncio.run_coroutine_threadsafe(
                    self.ollama_worker.message_obj.channel.send(message_text),
                    self.discord_bot.loop
                )
            # Optional: Send YouTube response to Discord channel (commented out)
            # elif self.ollama_worker.source == "youtube" and self.discord_bot and self.discord_bot.is_ready():
            #     channel_id = int(self.config["discord_channel_id"])
            #     channel = self.discord_bot.get_channel(channel_id)
            #     if channel:
            #         asyncio.run_coroutine_threadsafe(
            #             channel.send(f"[YouTube Response to {self.ollama_worker.username}]: {message_text}"),
            #             self.discord_bot.loop
            #         )
        self.current_assistant_message = None
        self.assistant_text_buffer = ""
        self.ollama_worker.quit()
        self.ollama_worker.wait()

    def handle_error(self, error_msg: str):
        self.add_message(f"Error: {error_msg}", is_user=False)
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.current_assistant_message = None
        self.assistant_text_buffer = ""
        self.ollama_worker.quit()
        self.ollama_worker.wait()

    def process_tts(self, text: str):
        try:
            text = text.lower()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            chunker = EnhancedTextChunker(max_chars=150)
            chunks = chunker.chunk_text(text)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                synthesis_result = get_tts_wav(
                    ref_wav_path=DEFAULT_PATHS["ref_audio"],
                    prompt_text=DEFAULT_PATHS["ref_text"],
                    prompt_language="English",
                    text=chunk,
                    text_language="English"
                )
                
                for sample_rate, audio_data in synthesis_result:
                    audio_data = audio_data.astype(np.float32)
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    self.audio_player.add_chunk(audio_data, sample_rate)
                
                self.audio_player.add_silence(0.15, sample_rate)
                
        except Exception as e:
            logging.error(f"TTS Error: {e}")

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.load_models()
            new_device = dialog.audio_device_combo.currentData()
            if new_device != self.audio_player.device:
                self.audio_player.set_device(new_device)
            self.config["discord_token"] = dialog.discord_token_edit.text().strip()
            self.config["discord_channel_id"] = dialog.discord_channel_edit.text().strip()
            self.config["youtube_url"] = dialog.youtube_url_edit.text().strip()
            save_config(self.config)

    def open_system_prompt(self):
        dialog = SystemPromptDialog(self)
        dialog.exec()

    def clear_chat(self):
        reply = QMessageBox.question(
            self, 'Clear Chat',
            'Are you sure you want to clear the current chat?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for i in reversed(range(self.chat_layout.count())):
                item = self.chat_layout.itemAt(i)
                if isinstance(item.widget(), QWidget):
                    item.widget().deleteLater()
            self.chat_layout.addStretch()
            self.current_assistant_message = None
            self.assistant_text_buffer = ""

    def clear_memory(self):
        reply = QMessageBox.question(
            self, 'Clear Memory',
            'Are you sure you want to clear all conversation memory?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.memory_store.clear_memory()
            QMessageBox.information(self, "Success", "Memory cleared successfully")

    def start_discord_chat(self):
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True

        self.discord_bot = commands.Bot(command_prefix="!", intents=intents)

        @self.discord_bot.event
        async def on_ready():
            self.bot_id = self.discord_bot.user.id
            logging.info(f"Discord bot logged in as {self.discord_bot.user}")
            channel_id = int(self.config["discord_channel_id"])
            channel = self.discord_bot.get_channel(channel_id)
            if channel:
                logging.info(f"Connected to Discord channel: {channel.name}")
            else:
                logging.error(f"Could not find Discord channel with ID: {channel_id}")
                self.discord_queue.put(("Error", "System", "discord", f"Invalid Discord channel ID: {channel_id}"))

        @self.discord_bot.event
        async def on_message(message):
            if not self.discord_running or message.author == self.discord_bot.user or message.channel.id != int(self.config["discord_channel_id"]):
                return
            
            username = message.author.name
            user_id = str(message.author.id)
            # Log all messages to memory
            self.memory_store.add_conversation_turn(user_id, username, "discord", message.content)
            self.add_message(f"[Discord] {username}: {message.content}", is_user=True, username=username)

            # Respond only to @pings or replies to bot
            mentions_bot = f"<@{self.bot_id}>" in message.content
            is_reply_to_bot = message.reference and message.reference.resolved and message.reference.resolved.author.id == self.bot_id
            if mentions_bot or is_reply_to_bot:
                self.discord_queue.put((user_id, username, "discord", message.content, message))

        token = self.config["discord_token"]
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.discord_bot.start(token))
        except Exception as e:
            logging.error(f"Failed to initialize Discord bot: {e}")
            self.discord_queue.put(("Error", "System", "discord", f"Failed to connect to Discord chat: {e}"))

    def stop_discord_chat(self):
        self.discord_running = False
        if self.discord_bot and self.discord_bot.is_ready():
            asyncio.run_coroutine_threadsafe(self.discord_bot.close(), self.discord_bot.loop)
        if self.discord_thread and self.discord_thread.is_alive():
            self.discord_thread.join(timeout=5)
        self.discord_timer.stop()

    def toggle_discord_chat(self):
        if self.discord_toggle_btn.isChecked():
            token = self.config["discord_token"]
            channel_id = self.config["discord_channel_id"]
            if not token or not channel_id:
                QMessageBox.warning(self, "Error", "Please configure Discord settings in the Settings dialog.")
                self.discord_toggle_btn.setChecked(False)
                return

            try:
                int(channel_id)
                self.discord_running = True
                self.discord_thread = threading.Thread(target=self.start_discord_chat, daemon=True)
                self.discord_thread.start()
                self.discord_timer.start(1000)
                self.discord_toggle_btn.setText("Stop Discord Chat")
            except ValueError:
                QMessageBox.warning(self, "Error", "Discord Channel ID must be a valid number.")
                self.discord_toggle_btn.setChecked(False)
        else:
            self.stop_discord_chat()
            self.discord_toggle_btn.setText("Start Discord Chat")

    def process_discord_messages(self):
        while not self.discord_queue.empty():
            user_id, username, source, message, message_obj = self.discord_queue.get()
            self.process_input(user_id, username, source, message, message_obj)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def start_youtube_chat(self):
        video_url = self.config["youtube_url"]
        video_id = self.extract_video_id(video_url)
        if not video_id:
            self.youtube_queue.put(("Error", "System", "youtube", "Invalid YouTube video URL."))
            return

        try:
            chat = pytchat.create(video_id=video_id, interruptable=False)
            if not chat.is_alive():
                raise ValueError("Failed to connect to YouTube chat. Please check the video ID.")

            while self.youtube_running and chat.is_alive():
                try:
                    for message in chat.get().sync_items():
                        if not self.youtube_running:
                            break
                        username = message.author.name
                        user_id = message.author.channelId
                        # Log all messages to memory
                        self.memory_store.add_conversation_turn(user_id, username, "youtube", message.message)
                        self.add_message(f"[YouTube] {username}: {message.message}", is_user=True, username=username)
                        # Process messages unless they start with //
                        if not message.message.strip().startswith('//'):
                            self.youtube_queue.put((user_id, username, "youtube", message.message))
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error fetching YouTube chat: {e}")
                    break
        except Exception as e:
            logging.error(f"Failed to initialize YouTube chat: {e}")
            self.youtube_queue.put(("Error", "System", "youtube", "Failed to connect to YouTube chat."))
            raise

    def stop_youtube_chat(self):
        self.youtube_running = False
        if self.youtube_thread and self.youtube_thread.is_alive():
            self.youtube_thread.join(timeout=5)
        self.youtube_timer.stop()

    def toggle_youtube_chat(self):
        if self.youtube_toggle_btn.isChecked():
            video_url = self.config["youtube_url"]
            if not video_url:
                QMessageBox.warning(self, "Error", "Please configure YouTube settings in the Settings dialog.")
                self.youtube_toggle_btn.setChecked(False)
                return

            try:
                self.youtube_running = True
                self.youtube_thread = threading.Thread(target=self.start_youtube_chat, daemon=True)
                self.youtube_thread.start()
                self.youtube_timer.start(1000)
                self.youtube_toggle_btn.setText("Stop YouTube Chat")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start YouTube chat: {e}")
                self.youtube_toggle_btn.setChecked(False)
        else:
            self.stop_youtube_chat()
            self.youtube_toggle_btn.setText("Start YouTube Chat")

    def extract_video_id(self, link):
        if "watch?v=" in link:
            return link.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in link:
            return link.split("youtu.be/")[1].split("?")[0]
        return None

    def process_youtube_messages(self):
        while not self.youtube_queue.empty():
            user_id, username, source, message = self.youtube_queue.get()
            self.process_input(user_id, username, source, message)

# System Prompt Dialog
class SystemPromptDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit System Prompt")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        self.setLayout(layout)

        header = QLabel("System Prompt")
        header.setStyleSheet(f"color: {THEME['text_primary']}; font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_PATHS["system_prompt"])
        layout.addWidget(self.prompt_edit)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        save_btn = QPushButton("Save")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.save_prompt)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {THEME['bg_secondary']}; }}
            QPlainTextEdit {{ background-color: {THEME['bg_tertiary']}; color: {THEME['text_primary']}; border: 1px solid {THEME['border']}; border-radius: 4px; padding: 8px; font-size: 14px; line-height: 1.4; }}
            QPushButton {{ background-color: {THEME['accent']}; color: {THEME['text_primary']}; border: none; border-radius: 4px; padding: 8px 16px; min-width: 100px; font-size: 14px; }}
            QPushButton:hover {{ background-color: {THEME['accent_hover']}; }}
        """)

    def save_prompt(self):
        DEFAULT_PATHS["system_prompt"] = self.prompt_edit.toPlainText()
        self.accept()

# Settings Dialog
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(800)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        self.setLayout(layout)

        tabs = QTabWidget()
        tabs.addTab(self.create_models_tab(), "Models")
        tabs.addTab(self.create_tts_tab(), "TTS Settings")
        tabs.addTab(self.create_audio_tab(), "Audio Settings")
        tabs.addTab(self.create_chat_tab(), "Chat Integrations")
        layout.addWidget(tabs)

        save_btn = QPushButton("Save Changes")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

    def create_models_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        model_group = QGroupBox("Model Paths")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(16)

        gpt_layout = QHBoxLayout()
        gpt_label = QLabel("GPT Model:")
        self.gpt_path_edit = QLineEdit()
        self.gpt_path_edit.setText(DEFAULT_PATHS["gpt_path"])
        gpt_browse_btn = QPushButton("Browse")
        gpt_browse_btn.clicked.connect(lambda: self.browse_file(self.gpt_path_edit))
        gpt_layout.addWidget(gpt_label)
        gpt_layout.addWidget(self.gpt_path_edit)
        gpt_layout.addWidget(gpt_browse_btn)
        model_layout.addLayout(gpt_layout)

        sovits_layout = QHBoxLayout()
        sovits_label = QLabel("SoVITS Model:")
        self.sovits_path_edit = QLineEdit()
        self.sovits_path_edit.setText(DEFAULT_PATHS["sovits_path"])
        sovits_browse_btn = QPushButton("Browse")
        sovits_browse_btn.clicked.connect(lambda: self.browse_file(self.sovits_path_edit))
        sovits_layout.addWidget(sovits_label)
        sovits_layout.addWidget(self.sovits_path_edit)
        sovits_layout.addWidget(sovits_browse_btn)
        model_layout.addLayout(sovits_layout)

        ref_layout = QHBoxLayout()
        ref_label = QLabel("Reference Audio:")
        self.ref_audio_edit = QLineEdit()
        self.ref_audio_edit.setText(DEFAULT_PATHS["ref_audio"])
        ref_browse_btn = QPushButton("Browse")
        ref_browse_btn.clicked.connect(lambda: self.browse_file(self.ref_audio_edit))
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.ref_audio_edit)
        ref_layout.addWidget(ref_browse_btn)
        model_layout.addLayout(ref_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        llm_group = QGroupBox("LLM Settings")
        llm_layout = QVBoxLayout()
        
        ollama_layout = QHBoxLayout()
        ollama_label = QLabel("Ollama Model:")
        self.ollama_model_edit = QLineEdit()
        self.ollama_model_edit.setText(DEFAULT_PATHS["ollama_model"])
        ollama_layout.addWidget(ollama_label)
        ollama_layout.addWidget(self.ollama_model_edit)
        llm_layout.addLayout(ollama_layout)
        
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_tts_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)
        
        settings = DEFAULT_PATHS["tts_settings"]
        
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QGridLayout()
        
        temp_label = QLabel("Temperature:")
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(settings["temperature"])
        voice_layout.addWidget(temp_label, 0, 0)
        voice_layout.addWidget(self.temp_spin, 0, 1)
        
        topk_label = QLabel("Top K:")
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(settings["top_k"])
        voice_layout.addWidget(topk_label, 1, 0)
        voice_layout.addWidget(self.topk_spin, 1, 1)

        topp_label = QLabel("Top P:")
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.1, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(settings["top_p"])
        voice_layout.addWidget(topp_label, 2, 0)
        voice_layout.addWidget(self.topp_spin, 2, 1)

        speed_label = QLabel("Speed Factor:")
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(settings["speed_factor"])
        voice_layout.addWidget(speed_label, 3, 0)
        voice_layout.addWidget(self.speed_spin, 3, 1)

        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)

        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout()

        batch_label = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(settings["batch_size"])
        advanced_layout.addWidget(batch_label, 0, 0)
        advanced_layout.addWidget(self.batch_spin, 0, 1)

        threshold_label = QLabel("Batch Threshold:")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(settings["batch_threshold"])
        advanced_layout.addWidget(threshold_label, 1, 0)
        advanced_layout.addWidget(self.threshold_spin, 1, 1)

        interval_label = QLabel("Fragment Interval:")
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 1.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(settings["fragment_interval"])
        advanced_layout.addWidget(interval_label, 2, 0)
        advanced_layout.addWidget(self.interval_spin, 2, 1)

        penalty_label = QLabel("Repetition Penalty:")
        self.penalty_spin = QDoubleSpinBox()
        self.penalty_spin.setRange(1.0, 2.0)
        self.penalty_spin.setSingleStep(0.05)
        self.penalty_spin.setValue(settings["repetition_penalty"])
        advanced_layout.addWidget(penalty_label, 3, 0)
        advanced_layout.addWidget(self.penalty_spin, 3, 1)

        self.split_bucket_check = QCheckBox("Split Bucket")
        self.split_bucket_check.setChecked(settings["split_bucket"])
        advanced_layout.addWidget(self.split_bucket_check, 4, 0, 1, 2)

        self.parallel_infer_check = QCheckBox("Parallel Inference")
        self.parallel_infer_check.setChecked(settings["parallel_infer"])
        advanced_layout.addWidget(self.parallel_infer_check, 5, 0, 1, 2)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)

        audio_group = QGroupBox("Audio Output")
        audio_layout = QVBoxLayout()

        self.audio_device_combo = QComboBox()
        self.audio_device_combo.addItem("Default", "default")
        for device in sd.query_devices():
            if device['max_output_channels'] > 0:
                self.audio_device_combo.addItem(device['name'], device['name'])
        
        saved_device = DEFAULT_PATHS.get("audio_output_device", "default")
        index = self.audio_device_combo.findData(saved_device)
        if index >= 0:
            self.audio_device_combo.setCurrentIndex(index)
        
        audio_layout.addWidget(QLabel("Output Device:"))
        audio_layout.addWidget(self.audio_device_combo)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)

        discord_group = QGroupBox("Discord Settings")
        discord_layout = QVBoxLayout()
        self.discord_token_edit = QLineEdit()
        self.discord_token_edit.setPlaceholderText("Enter Discord Bot Token")
        self.discord_token_edit.setText(self.parent().config.get("discord_token", ""))
        discord_layout.addWidget(QLabel("Bot Token:"))
        discord_layout.addWidget(self.discord_token_edit)
        self.discord_channel_edit = QLineEdit()
        self.discord_channel_edit.setPlaceholderText("Enter Discord Channel ID")
        self.discord_channel_edit.setText(self.parent().config.get("discord_channel_id", ""))
        discord_layout.addWidget(QLabel("Channel ID:"))
        discord_layout.addWidget(self.discord_channel_edit)
        discord_group.setLayout(discord_layout)
        layout.addWidget(discord_group)

        youtube_group = QGroupBox("YouTube Settings")
        youtube_layout = QVBoxLayout()
        self.youtube_url_edit = QLineEdit()
        self.youtube_url_edit.setPlaceholderText("Enter YouTube Live URL")
        self.youtube_url_edit.setText(self.parent().config.get("youtube_url", ""))
        youtube_layout.addWidget(QLabel("Live URL:"))
        youtube_layout.addWidget(self.youtube_url_edit)
        youtube_group.setLayout(youtube_layout)
        layout.addWidget(youtube_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{ background-color: {THEME['bg_secondary']}; }}
            QGroupBox {{ color: {THEME['text_primary']}; font-weight: bold; border: 1px solid {THEME['border']}; border-radius: 4px; margin-top: 12px; padding-top: 20px; }}
            QLabel {{ color: {THEME['text_primary']}; }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{ background-color: {THEME['bg_tertiary']}; color: {THEME['text_primary']}; border: 1px solid {THEME['border']}; border-radius: 4px; padding: 4px 8px; }}
            QCheckBox {{ color: {THEME['text_primary']}; }}
            QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid {THEME['border']}; border-radius: 3px; background-color: {THEME['bg_tertiary']}; }}
            QCheckBox::indicator:checked {{ background-color: {THEME['accent']}; }}
            QPushButton {{ background-color: {THEME['accent']}; color: {THEME['text_primary']}; border: none; border-radius: 4px; padding: 8px 16px; min-width: 100px; }}
            QPushButton:hover {{ background-color: {THEME['accent_hover']}; }}
            QTabWidget::pane {{ border: 1px solid {THEME['border']}; border-radius: 4px; background-color: {THEME['bg_secondary']}; }}
            QTabBar::tab {{ background-color: {THEME['bg_tertiary']}; color: {THEME['text_secondary']}; border: 1px solid {THEME['border']}; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
            QTabBar::tab:selected {{ background-color: {THEME['accent']}; color: {THEME['text_primary']}; }}
        """)

    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            line_edit.setText(file_path)

    def save_settings(self):
        DEFAULT_PATHS["gpt_path"] = self.gpt_path_edit.text()
        DEFAULT_PATHS["sovits_path"] = self.sovits_path_edit.text()
        DEFAULT_PATHS["ref_audio"] = self.ref_audio_edit.text()
        DEFAULT_PATHS["ollama_model"] = self.ollama_model_edit.text()

        tts_settings = DEFAULT_PATHS["tts_settings"]
        tts_settings.update({
            "temperature": self.temp_spin.value(),
            "top_k": self.topk_spin.value(),
            "top_p": self.topp_spin.value(),
            "speed_factor": self.speed_spin.value(),
            "batch_size": self.batch_spin.value(),
            "batch_threshold": self.threshold_spin.value(),
            "fragment_interval": self.interval_spin.value(),
            "repetition_penalty": self.penalty_spin.value(),
            "split_bucket": self.split_bucket_check.isChecked(),
            "parallel_infer": self.parallel_infer_check.isChecked()
        })

        DEFAULT_PATHS["audio_output_device"] = self.audio_device_combo.currentData()
        DEFAULT_PATHS["discord_token"] = self.discord_token_edit.text().strip()
        DEFAULT_PATHS["discord_channel_id"] = self.discord_channel_edit.text().strip()
        DEFAULT_PATHS["youtube_url"] = self.youtube_url_edit.text().strip()

        save_config(DEFAULT_PATHS)
        self.accept()

# Main Application
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    app.aboutToQuit.connect(window.closeEvent)
    
    sys.exit(app.exec())