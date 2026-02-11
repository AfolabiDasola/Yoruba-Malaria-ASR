# 🦟 Yoruba Malaria ASR (Wav2Vec2 + KenLM)

This project implements an Automatic Speech Recognition (ASR) system tailored for **Yoruba medical speech**, specifically focusing on malaria diagnosis.

## 🔗 Live Demo
**Try the model here:** [Hugging Face Space](https://huggingface.co/spaces/AfolabiDasola/YorMal)  
*(Click the link to record your voice and test the transcription!)*

## 🎯 Project Goal
Malaria is a significant health challenge in Nigeria. Language barriers often hinder effective diagnosis. This model aims to transcribe Yoruba descriptions of symptoms (e.g., *'ibà'*, *'orí fífọ'*) to assist healthcare workers.

## 🛠️ Tech Stack
* **Model:** Fine-tuned `facebook/wav2vec2-large-xlsr-53`
* **Language Model:** 3-gram KenLM (trained on Yoruba medical corpus)
* **Processor:** PyCTCDecode with KenLM integration
* **Deployment:** Hugging Face Spaces (Gradio)

## 📂 Repository Structure
* `Training_Notebook.ipynb`: The complete code used to fine-tune the model.
* `app.py`: The deployment script running on Hugging Face.
* `requirements.txt`: Dependencies required to run the inference.

## 🚀 How to Run Locally
```python
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
import librosa
import torch

# Load the model
model_id = "AfolabiDasola/yoruba-malaria-asr-with-lm"
processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

# Load audio
audio, rate = librosa.load("my_audio.wav", sr=16000)

# Transcribe
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
logits = model(inputs.input_values).logits
transcription = processor.decode(logits.cpu().numpy()[0], beam_width=50).text

print(transcription)
