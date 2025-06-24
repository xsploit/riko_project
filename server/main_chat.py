from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio

import os
import time
### transcribe audio 
import uuid
import soundfile as sf


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


print(' \n ========= Starting Chat... ================ \n')
whisper_model = WhisperModel("base.en", device="cuda", compute_type="float16")



while True:

    conversation_recording = "/home/rayenfeng/riko_v1/conversation.wav"

    # record_on_speech(
    #         output_file=conversation_recording,
    #         samplerate=44100,
    #         channels=1,
    #         silence_threshold=0.02,  # Adjust based on your microphone sensitivity
    #         silence_duration=1,     # Stop after 3 seconds of silence
    #         device=None             # Use default device, or specify by ID or name
    #     )

    # user_spoken_text = transcribe_audio(whisper_model, aud_path=conversation_recording)

    user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)
    # stop squence 

    if user_spoken_text.lower().strip() in ["exit.", "quit."]:
        print("Goodbye!")
        break

    ### pass to LLM and get a LLM output.

    llm_output = llm_response(user_spoken_text)

    tts_read_text = llm_output

    ### file organization 

    # 1. Generate a unique filename
    uid = uuid.uuid4().hex
    filename = f"output_{uid}.wav"
    output_wav_path = f"./audio/{filename}"


    # generate audio and save it to client/audio 
    gen_aud_path = sovits_gen(tts_read_text,output_wav_path)


    play_audio(output_wav_path)
    # # Example
    # duration = get_wav_duration(output_wav_path)

    # print("waiting for audio to finish...")
    # time.sleep(duration)