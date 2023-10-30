from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
# import soundfile as sf

import librosa

SAMPLERATE = 16000


def speech_to_text(audio_file):
    # data, samplerate = sf.read(audio_file)
    data, samplerate = librosa.load(audio_file, sr=SAMPLERATE)

    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

    inputs = processor(data, sampling_rate=samplerate, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription


if __name__ == "__main__":
    speech_to_text('REC_20231021115031253.flac')
