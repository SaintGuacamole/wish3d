from fastapi import FastAPI, File, UploadFile

from speech_to_text import speech_to_text
from text_to_3d import get_latents, decode_latents_to_files, create_zipped_response

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello"}


@app.get("/text_to_obj")
def shape_get(prompt: str, nr_samples: int = 3):

    _latents = get_latents(prompt, nr_samples)
    _filenames = decode_latents_to_files(_latents)
    return create_zipped_response(_filenames)


@app.post("/audio_to_text")
def voice_to_text(audio: UploadFile):
    text = speech_to_text(audio.file)
    return text


@app.post("/audio_to_obj")
def voice_to_text(audio: UploadFile, nr_samples: int = 3):
    text = speech_to_text(audio.file)[0]

    _latents = get_latents(text, nr_samples)
    _filenames = decode_latents_to_files(_latents)
    return create_zipped_response(_filenames)

