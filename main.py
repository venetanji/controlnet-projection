from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import os
from PIL import Image

controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16, controlnet=controlnet_canny
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to("cuda")

canny = Image.open("canny.png")
# depth = load_image("depth.png")

idx = 0
while True:
    
    image = pipe(
            prompt="best quality, extremely detailed, underwater, underwater scene, underwater world, underwater photography",
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            image=canny,
            num_inference_steps=20,
            width=512,
            height=512,
    ).images[0]
    # save frames zero padded to 6 digits in frames folder
    image.save(os.path.join("frames", f"{idx:06d}.png"))
    idx+=1
    