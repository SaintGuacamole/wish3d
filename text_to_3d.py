import os
import zipfile
from io import StringIO, BytesIO
from typing import List

import torch
from fastapi.responses import Response

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

GUIDANCE_SCALE = 15.
ZIP_SUB_DIRECTORY = "archive"

print("If running for the first time, model download will take several minutes")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transmitter = load_model('transmitter', device=device)
text_to_geom_model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

print("Models are loaded, API available momentarily")


def get_latents(prompt: str, nr_samples: int):
    _prompt = [prompt] * nr_samples
    _batch_size = len(_prompt)

    _latents = sample_latents(
        batch_size=_batch_size,
        model=text_to_geom_model,
        diffusion=diffusion,
        guidance_scale=GUIDANCE_SCALE,
        model_kwargs=dict(texts=_prompt),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=32,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    return _latents


def decode_latents_to_files(latents: torch.Tensor) -> List:
    filenames = []
    for _i, _latent in enumerate(latents):
        with open(f'sample_{_i}.obj', 'w') as f:
            t = decode_latent_mesh(transmitter, _latent).tri_mesh()
            t.write_obj(f)
        f.close()
        filenames.append(f'mesh_{_i}.obj')

    return filenames


def create_zipped_response(filenames):
    # Open StringIO to grab in-memory ZIP contents
    s = BytesIO()
    # The zip compressor
    zf = zipfile.ZipFile(s, "w")

    for fpath in filenames:
        print(fpath)
        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)
        zip_path = os.path.join(ZIP_SUB_DIRECTORY, fname)

        # Add file, at correct path
        zf.write(fpath, zip_path)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed")
    # ..and correct content-disposition

    return resp
