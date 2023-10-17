import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

from fastapi import FastAPI

print("If running for the first time, model download will take several minutes")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transmitter = load_model('transmitter', device=device)
text_to_geom_model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

print("Models are loaded, API available momentarily")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/shape")
async def shape_get(prompt: str, nr_samples: int = 3):
    _guidance_scale = 15.0
    _prompt = [prompt] * nr_samples
    _batch_size = len(_prompt)

    _latents = sample_latents(
        batch_size=_batch_size,
        model=text_to_geom_model,
        diffusion=diffusion,
        guidance_scale=_guidance_scale,
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

    _data = []
    for _i, _latent in enumerate(_latents):
        _p = decode_latent_mesh(transmitter, _latent)
        _data.append({
            "vertices": _p.verts.detach().cpu().clone().tolist(),
            "faces": _p.faces.detach().cpu().clone().tolist(),
            "color": {
                "R": _p.vertex_channels["R"].detach().cpu().clone().tolist(),
                "G": _p.vertex_channels["G"].detach().cpu().clone().tolist(),
                "B": _p.vertex_channels["B"].detach().cpu().clone().tolist()
            }
        })
    return _data
