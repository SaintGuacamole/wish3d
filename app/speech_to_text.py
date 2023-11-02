import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


SAMPLERATE = 16000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Setting up whisper, device: {device}")

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", device=device)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to(device)
model.config.forced_decoder_ids = None

print("whisper setup done")


def whisper_to_text(audio_file):
    with torch.no_grad():
        # load model and processor
        data, samplerate = librosa.load(audio_file, sr=SAMPLERATE)
        input_features = processor(data, sampling_rate=SAMPLERATE, return_tensors="pt").input_features

        predicted_ids = model.generate(input_features.to(device))

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

#
# def speech_to_text(audio_file):
#     # data, samplerate = sf.read(audio_file)
#     data, samplerate = librosa.load(audio_file, sr=SAMPLERATE)
#
#     model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
#     processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
#
#     inputs = processor(data, sampling_rate=samplerate, return_tensors="pt")
#     generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
#
#     transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
#     return transcription
#
#
# if __name__ == "__main__":
#     speech_to_text('REC_20231021115031253.flac')
