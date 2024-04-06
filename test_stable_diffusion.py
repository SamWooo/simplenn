import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


# 组合图像，生成grid
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# 加载文生图pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # 或者使用 SD v1.4: "CompVis/stable-diffusion-v1-4"
    torch_dtype=torch.float16
).to("cuda")

# 输入text，这里text又称为prompt
prompts = [
    "a photograph of an astronaut riding a horse",
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    "An avocado armchair",
    "A white dog wearing sunglasses"
]

generator = torch.Generator("cuda").manual_seed(42)  # 定义随机seed，保证可重复性

# 执行推理
images = pipe(
    prompts,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    generator=generator
).images

grid = image_grid(images, rows=1, cols=4)
grid