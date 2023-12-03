
import json
import os

import uuid

import numpy as np
from fastapi import FastAPI, UploadFile, Response
from starlette.responses import FileResponse

from speech_to_text import whisper_to_text
from text_to_3d import get_latents, decode_latents_to_files, create_zipped_response, decode_latents_to_mesh, \
    decode_single_file, decode_dict

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
def voice_to_text(
        audio: UploadFile,
        nr_samples: int = 1,
        target_nr_faces: int = 1000,
        karras_steps: int = 32,
        sigma_min: float = 1e-3,
        sigma_max: float = 160.
):
    task_id = uuid.uuid4()
    os.mkdir('F:/wish3d/' + str(task_id))

    # if nr_samples > 1:
    #     return Response("can't generate more than 1 file", status_code=400)
    print(f"Start, requested {nr_samples} sample{'s' if nr_samples > 1 else ''}")
    text = whisper_to_text(audio.file)[0]

    # if model == "zero":
    #     return Response("feature turned off", 400)
        # zero_one_two_three(text, task_id)
        #
        # mesh_path = one_two_three_four_five(task_id)
        #
        # return FileResponse(mesh_path, media_type="text/plain")

    _latents = get_latents(text, nr_samples, karras_steps, sigma_min, sigma_max)
    # obj = decode_single_file(_latents, f"F:/wish3d/{str(task_id)}/" + text + ".obj")

    # return FileResponse(obj, media_type="text/plain")

    meshes = decode_dict(_latents, task_id, target_nr_faces)
    # return Response(obj, media_type="text/plain")
    # meshes = decode_latents_to_mesh(_latents)

    # print(meshes.keys())
    return json.dumps(meshes, cls=NumpyEncoder)

