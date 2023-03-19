from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import os
from PIL import Image
import cv2
import numpy as np

controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
	"runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16, controlnet=controlnet_canny
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to("cuda")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# canny = Image.open("canny.png")
# depth = load_image("depth.png")

idx = 0
while True:    
    ret, frame = cap.read()
    canny = cv2.Canny(frame, 50, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    cv2.imshow("frame", canny)
    cv2.waitKey(1)
    canny = Image.fromarray(canny)


    image = pipe(
            prompt="best quality, extremely detailed, underwater, underwater scene, underwater world, underwater photography",
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            image=canny,
            num_inference_steps=20,
            width=640,
            height=360,
    ).images[0]
    cv2.imshow("gen", np.array(image))
    cv2.waitKey(1)
    # save frames zero padded to 6 digits in frames folder
    image.save(os.path.join("frames", f"{idx:06d}.png"))
    idx+=1
    