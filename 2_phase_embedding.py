import sounddevice as sd
import numpy as np
import queue
import time
import json
from collections import deque
from datetime import datetime

import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel


SAMPLE_RATE = 16000
BLOCK_SIZE = 4000       # ~250ms blocks from sounddevice
MIC_INDEX = 0           # adjust if needed (sd.query_devices() to check)
PROCESS_SECONDS = 4     # how many seconds we send to Whisper each chunk
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model - multilingual MiniLM style (we'll use the HF checkpoint directly)
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Note: using AutoModel + AutoTokenizer to avoid sentence-transformers dependency

# Path to write logs (turn text + embedding)
OUTPUT_LOG = "turns_embeddings.jsonl"

# How many previous turns to keep (context window)
CONTEXT_TURNS = 8
# ----------------------------------------

# Thread-safe audio queue used by callback
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        # Print status if there are issues
        print("Audio status:", status)
    audio_queue.put(indata.copy())

# ---------------- ASR MODEL ----------------
print("Loading Whisper model (may take a moment)...")
whisper_model = WhisperModel("small", device=DEVICE, compute_type="int8")  # change if you want medium/large
print("Whisper loaded.")

# ---------------- EMBEDDING MODEL (transformers AutoModel) ----------------
print("Loading tokenizer and embedding model (transformers)...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True)
auto_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
auto_model.to(DEVICE)
auto_model.eval()
print("Embedding model loaded on", DEVICE)

# Mean-pooling helper (like sentence-transformers)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embedding(text):
    # Tokenize + run through AutoModel, then mean-pool and L2-normalize
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_output = auto_model(**encoded_input)
    pooled = mean_pooling(model_output, encoded_input['attention_mask'])  # (1, hidden)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.squeeze(0).cpu().numpy()  # return 1D numpy array

# ---------------- Context buffer & logging ----------------
turn_buffer = deque(maxlen=CONTEXT_TURNS)        # store (text, embedding, timestamp)
print("Context buffer initialized (size {})".format(CONTEXT_TURNS))

# Ensure output log exists
open(OUTPUT_LOG, "a").close()

# ---------------- Main real-time loop ----------------
def main():
    print("Starting audio stream...")
    stream = sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()
    print("🎙️ Listening... Speak clearly (Ctrl+C to stop).")

    buffer = np.zeros((0,), dtype=np.float32)
    samples_per_chunk = SAMPLE_RATE * PROCESS_SECONDS

    try:
        while True:
            # Block until at least one block is in queue
            audio_block = audio_queue.get()
            buffer = np.concatenate((buffer, audio_block[:, 0]))

            if len(buffer) >= samples_per_chunk:
                proc_chunk = buffer[:samples_per_chunk]
                buffer = buffer[samples_per_chunk:]

                # Run Whisper transcription on the chunk
                segments, info = whisper_model.transcribe(
                    proc_chunk,
                    language=None,    # autodetect (keeps Hinglish)
                    vad_filter=True,
                    beam_size=5,
                )

                # segments is iterable of segment objects with .start/.end/.text
                for seg in segments:
                    turn_text = seg.text.strip()
                    if not turn_text:
                        continue

                    # Get embedding for this turn (semantic vector)
                    try:
                        emb = get_embedding(turn_text)  # 1D numpy array
                    except Exception as e:
                        print("Embedding error:", e)
                        emb = None

                    ts = datetime.utcnow().isoformat() + "Z"
                    turn_buffer.append((turn_text, emb, ts))

                    # Print the turn and info
                    print("\n🗣️ Heard:", turn_text)
                    if emb is not None:
                        print("🧠 Embedding shape:", emb.shape)
                    else:
                        print("🧠 Embedding: ERROR")

                    # Save to log for later model training
                    record = {
                        "timestamp": ts,
                        "text": turn_text,
                        "embedding": emb.tolist() if emb is not None else None
                    }
                    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # *** SIMPLE DEMO: print context summary (texts only) ***
                    context_texts = [t for (t, e, ts) in turn_buffer]
                    print("📚 Context (last {} turns):".format(len(context_texts)))
                    for i, cx in enumerate(context_texts, 1):
                        print("  {}: {}".format(i, cx))

                # end for segments

            # small sleep to avoid busy waiting
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Stopping.")
        stream.stop()

if __name__ == "__main__":
    main()