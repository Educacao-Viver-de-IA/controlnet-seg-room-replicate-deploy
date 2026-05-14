"""
ControlNet Seg Room — gerador de imagens de quartos a partir de mapas de segmentação.
Base: BertChristiaens/controlnet-seg-room + stable-diffusion-v1-5.
Input: segmentation map (RGB) + prompt textual.
Output: imagem gerada do quarto.
"""
import json
import os
import sys
import tempfile
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()
import torch
print(f"[module] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
from PIL import Image
from cog import BasePredictor, Input, Path
print(f"[module] cog+PIL OK", flush=True)
sys.stdout.flush()

CONTROLNET_PATH = "/src/weights/controlnet-seg-room"
SD_BASE_PATH = "/src/weights/stable-diffusion-v1-5"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.pipe = None
        self.setup_error = None
        try:
            from diffusers import (
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                UniPCMultistepScheduler,
            )
            print(f"[setup] diffusers imported, loading ControlNet from {CONTROLNET_PATH}", flush=True)
            sys.stdout.flush()
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_PATH, local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            print(f"[setup] ControlNet loaded ({time.time()-t0:.1f}s), loading SD pipeline...", flush=True)
            sys.stdout.flush()

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                SD_BASE_PATH,
                controlnet=controlnet,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                # Memory optimization
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass

            print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

    def predict(
        self,
        seg_image: Path = Input(
            description="Segmentation map RGB (ADE20K palette). Pixels coloridos representam tipos de objetos.",
        ),
        prompt: str = Input(
            description="Descrição textual do quarto desejado.",
            default="a luxurious modern living room with large windows, natural light, scandinavian style",
        ),
        negative_prompt: str = Input(
            description="Coisas a evitar.",
            default="ugly, blurry, low quality, distorted, deformed, dark, gloomy",
        ),
        num_inference_steps: int = Input(
            description="Passos de denoising. Mais = melhor qualidade, mais lento.",
            default=20, ge=10, le=100,
        ),
        guidance_scale: float = Input(
            description="CFG scale (quanto seguir o prompt).",
            default=7.5, ge=1.0, le=20.0,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Quanto seguir a segmentação (0 = ignora seg, 2 = seg forte).",
            default=1.0, ge=0.0, le=2.0,
        ),
        seed: int = Input(default=-1, description="Seed (-1 = random)."),
        width: int = Input(default=512, ge=256, le=1024),
        height: int = Input(default=512, ge=256, le=1024),
    ) -> Path:
        if self.pipe is None:
            raise RuntimeError(f"Modelo não carregou: {getattr(self, 'setup_error', '?')}")

        t0 = time.time()
        seg = Image.open(seg_image).convert("RGB").resize((width, height), Image.NEAREST)

        if seed < 0:
            generator = None
        else:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

        print(f"[predict] generating {width}x{height} steps={num_inference_steps}", flush=True)
        result = self.pipe(
            prompt=prompt,
            image=seg,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            width=width,
            height=height,
        )
        out_img = result.images[0]

        out_dir = tempfile.mkdtemp()
        out_path = os.path.join(out_dir, "output.png")
        out_img.save(out_path)
        print(f"[predict] DONE ({time.time()-t0:.1f}s)", flush=True)
        return Path(out_path)
