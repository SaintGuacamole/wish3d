import io
import json
import time

import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response
from starlette.responses import JSONResponse

from speech_to_text import whisper_to_text
from text_to_3d import get_latents, decode_latents_to_files, create_zipped_response, decode_latents_to_mesh

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
    text = whisper_to_text(audio.file)[0]
    return text


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.post("/audio_to_obj")
def voice_to_text(audio: UploadFile, nr_samples: int = 1):
    print(f"Start, requested {nr_samples} sample{'s' if nr_samples > 1 else ''}")
    begin = time.time()
    text = whisper_to_text(audio.file)[0]

    print(f"Text at {time.time() - begin} s, {text}")

    _latents = get_latents(text, nr_samples)
    meshes = decode_latents_to_mesh(_latents)

    print(meshes.keys())
    return json.dumps(meshes, cls=NumpyEncoder)
    # _filenames = decode_latents_to_files(_latents)
    # return create_zipped_response(_filenames)

