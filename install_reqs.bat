@echo off
REM CHECK NVIDIA VERSION WITH NVIDIA-SMI I HAVE 12.7 BUT IF YOU HAVE 12.8 USE: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Setup Visual Studio build environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

pip install uv
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -r extra-req.txt --no-deps

REM Install requirements without pyopenjtalk first
pip install numpy scipy tensorboard "librosa==0.10.2" numba "pytorch-lightning>=2.4" "gradio>5" ffmpeg-python onnxruntime-gpu tqdm "funasr==1.0.27" cn2an pypinyin g2p_en torchaudio "modelscope==1.10.0" sentencepiece "transformers>=4.43" peft chardet PyYAML psutil jieba_fast jieba split-lang "fast_langdetect>=0.3.1" wordsegment rotary_embedding_torch ToJyutping g2pk2 ko_pron "fastapi[standard]>=0.115.2" x_transformers "torchmetrics<=1.5" "pydantic<=2.10.6" "ctranslate2>=4.0,<5" "huggingface_hub>=0.13" "tokenizers>=0.13,<1" "av>=11" openai

REM Try pyopenjtalk with conda if available, otherwise skip
conda install pyopenjtalk -c conda-forge -y 2>nul || echo "Skipping pyopenjtalk - install Visual Studio Build Tools if needed"

pip install nltk
python -c "import nltk; [nltk.download(pkg) for pkg in ['averaged_perceptron_tagger', 'cmudict']]"

pause