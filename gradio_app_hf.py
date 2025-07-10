from utils import offload
from utils.offload import profile_type

import os
import sys
import torch
import diffusers
import transformers
import argparse
import peft
import copy
import cv2
import gc
import tempfile
import imageio
import threading
import gradio as gr
import numpy as np

from flash_attn import flash_attn_func

from peft import LoraConfig
from omegaconf import OmegaConf
from safetensors.torch import safe_open
from PIL import Image, ImageDraw, ImageFilter
from huggingface_hub import hf_hub_download
from transformers import pipeline

from models import HunyuanVideoTransformer3DModel
from pipelines import HunyuanVideoImageToVideoPipeline

header = """
# DRA-Ctrl Gradio App

<div style="text-align: center; display: flex; justify-content: left; gap: 5px;">
<a href="https://arxiv.org/pdf/2505.23325"><img src="https://img.shields.io/badge/ariXv-Paper-A42C25.svg" alt="arXiv"></a>
<a href="https://arxiv.org/abs/2505.23325"><img src="https://img.shields.io/badge/ariXv-Page-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/Kunbyte/DRA-Ctrl"><img src="https://img.shields.io/badge/🤗-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/spaces/Kunbyte/DRA-Ctrl"><img src="https://img.shields.io/badge/🤗-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://github.com/Kunbyte-AI/DRA-Ctrl"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github&" alt="GitHub"></a>
<a href="https://dra-ctrl-2025.github.io/DRA-Ctrl/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project"></a>
</div>
"""

notice = """
For easier testing, in spatially-aligned image generation tasks, when passing the condition image to `gradio_app`, 
there's no need to manually input edge maps, depth maps, or other condition images - only the original image is required. 
The corresponding condition images will be automatically extracted.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="DRA-Ctrl Gradio App")
    parser.add_argument(
        "--vram_optimization",
        type=str,
        required=True,
        default="No_Optimization",
        help="VRAM optimization strategy. (Required, default: NO_OPTIMIZATION)"
    )
    args = parser.parse_args()

    vram_optimization_opts = [
        'No_Optimization',
        'HighRAM_HighVRAM',
        'HighRAM_LowVRAM',
        'LowRAM_HighVRAM',
        'LowRAM_LowVRAM',
        'VerylowRAM_LowVRAM'
    ]
    if args.vram_optimization not in vram_optimization_opts:
        raise ValueError(
            f"Invalid vram_optimization: {args.vram_optimization}. "
            f"Must be one of {vram_optimization_opts}"
        )

    return args.vram_optimization

def init_basemodel(vram_optimization):
    global transformer, scheduler, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, image_processor, pipe, current_task
    pipe = None
    current_task = None

    # init models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.bfloat16
    i2v_model_root = '/data/home/caohengyuan/iLego_chy/cache/HunyuanVideo-I2V' # 'ckpts/HunyuanVideo-I2V'
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(f'{i2v_model_root}/transformer', 
                                                                 inference_subject_driven=False, 
                                                                 low_cpu_mem_usage=True, 
                                                                 torch_dtype=weight_dtype).requires_grad_(False)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
    vae = diffusers.AutoencoderKLHunyuanVideo.from_pretrained(f'{i2v_model_root}/vae', 
                                                              low_cpu_mem_usage=True, 
                                                              torch_dtype=weight_dtype).requires_grad_(False)
    text_encoder = transformers.LlavaForConditionalGeneration.from_pretrained(f'{i2v_model_root}/text_encoder', 
                                                                              low_cpu_mem_usage=True, 
                                                                              torch_dtype=weight_dtype).requires_grad_(False)
    text_encoder_2 = transformers.CLIPTextModel.from_pretrained(f'{i2v_model_root}/text_encoder_2', 
                                                                low_cpu_mem_usage=True, 
                                                                torch_dtype=weight_dtype).requires_grad_(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f'{i2v_model_root}/tokenizer')
    tokenizer_2 = transformers.CLIPTokenizer.from_pretrained(f'{i2v_model_root}/tokenizer_2')
    image_processor = transformers.CLIPImageProcessor.from_pretrained(f'{i2v_model_root}/image_processor')

    vae.enable_tiling()
    vae.enable_slicing()

    # insert LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=[
            'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
            'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
            'ff.net.0.proj', 'ff.net.2',
            'ff_context.net.0.proj', 'ff_context.net.2',
            'norm1_context.linear', 'norm1.linear',
            'norm.linear', 'proj_mlp', 'proj_out',
        ]
    )
    transformer.add_adapter(lora_config)

    # hack LoRA forward
    def create_hacked_forward(module):
        if not hasattr(module, 'original_forward'):
            module.original_forward = module.forward
        img_sequence_length = int((512 / 8 / 2) ** 2)
        encoder_sequence_length = 144 + 252 # encoder sequence: 144 img 252 txt
        num_imgs = 4
        num_generated_imgs = 3

        def hacked_lora_forward(self, x, *args, **kwargs):
            lora_forward = self.original_forward

            if x.shape[1] == img_sequence_length * num_imgs and len(x.shape) > 2:
                return torch.cat((
                    lora_forward(x[:, :-img_sequence_length*num_generated_imgs], *args, **kwargs),
                    self.base_layer(x[:, -img_sequence_length*num_generated_imgs:], *args, **kwargs)
                ), dim=1)
            elif x.shape[1] == encoder_sequence_length * 2 or x.shape[1] == encoder_sequence_length:
                return lora_forward(x, *args, **kwargs)
            elif x.shape[1] == img_sequence_length * num_imgs + encoder_sequence_length:
                return torch.cat((
                    lora_forward(x[:, :(num_imgs - num_generated_imgs)*img_sequence_length], *args, **kwargs),
                    self.base_layer(x[:, (num_imgs - num_generated_imgs)*img_sequence_length:-encoder_sequence_length], *args, **kwargs),
                    lora_forward(x[:, -encoder_sequence_length:], *args, **kwargs)
                ), dim=1)
            elif x.shape[1] == img_sequence_length * num_imgs + encoder_sequence_length * 2:
                return torch.cat((
                    lora_forward(x[:, :(num_imgs - num_generated_imgs)*img_sequence_length], *args, **kwargs),
                    self.base_layer(x[:, (num_imgs - num_generated_imgs)*img_sequence_length:-2*encoder_sequence_length], *args, **kwargs),
                    lora_forward(x[:, -2*encoder_sequence_length:], *args, **kwargs)
                ), dim=1)
            elif x.shape[1] == 3072:
                return self.base_layer(x, *args, **kwargs)
            else:
                raise ValueError(
                    f"hacked_lora_forward receives unexpected sequence length: {x.shape[1]}, input shape: {x.shape}!"
                )

        return hacked_lora_forward.__get__(module, type(module))

    for n, m in transformer.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)

    pipe = HunyuanVideoImageToVideoPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=copy.deepcopy(scheduler),
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        image_processor=image_processor,
    )

    if vram_optimization == 'No_Optimization':
        pipe.to(device)
    else:
        [
        'No_Optimization',
        'HighRAM_HighVRAM',
        'HighRAM_LowVRAM',
        'LowRAM_HighVRAM',
        'LowRAM_LowVRAM',
        'VerylowRAM_LowVRAM'
    ]
        if vram_optimization == 'HighRAM_HighVRAM':
            optimization_type = profile_type.HighRAM_HighVRAM
        elif vram_optimization == 'HighRAM_HighVRAM':
            optimization_type = profile_type.HighRAM_HighVRAM
        elif vram_optimization == 'HighRAM_LowVRAM':
            optimization_type = profile_type.HighRAM_LowVRAM
        elif vram_optimization == 'LowRAM_HighVRAM':
            optimization_type = profile_type.LowRAM_HighVRAM
        elif vram_optimization == 'LowRAM_LowVRAM':
            optimization_type = profile_type.LowRAM_LowVRAM
        elif vram_optimization == 'VerylowRAM_LowVRAM':
            optimization_type = profile_type.VerylowRAM_LowVRAM
        offload.profile(pipe, optimization_type)

def process_image_and_text(condition_image, target_prompt, condition_image_prompt, task, random_seed, num_steps, inpainting, fill_x1, fill_x2, fill_y1, fill_y2):
    # set up the model
    global pipe, current_task, transformer
    if current_task != task:
        # load LoRA weights
        model_root = f'ckpts/{task}.safetensors'
        try:
            with safe_open(model_root, framework="pt") as f:
                lora_weights = {}
                for k in f.keys():
                    param = f.get_tensor(k) 
                    if k.endswith(".weight"):
                        k = k.replace('.weight', '.default.weight')
                    lora_weights[k] = param
                transformer.load_state_dict(lora_weights, strict=False)
        except Exception as e:
            raise ValueError(f'{e}')

        transformer.requires_grad_(False)

    # start generation
    c_txt = None if condition_image_prompt == "" else condition_image_prompt
    c_img = condition_image.resize((512, 512))
    t_txt = target_prompt

    if task not in ['subject_driven', 'style_transfer']:
        if task == "canny":
            def get_canny_edge(img):
                img_np = np.array(img)
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
                edges_tmp = Image.fromarray(edges).convert("RGB")
                edges[edges == 0] = 128
                return Image.fromarray(edges).convert("RGB")
            c_img = get_canny_edge(c_img)
        elif task == "coloring":
            c_img = (
                c_img.resize((512, 512))
                .convert("L")
                .convert("RGB")
            )
        elif task == "deblurring":
            blur_radius = 10
            c_img = (
                c_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .resize((512, 512))
                .convert("RGB")
            )
        elif task == "depth":
            def get_depth_map(img):
                from transformers import pipeline

                depth_pipe = pipeline(
                    task="depth-estimation",
                    model="ckpts/depth-anything-small-hf",
                    device="cpu",
                )
                return depth_pipe(img)["depth"].convert("RGB").resize((512, 512))
            c_img = get_depth_map(c_img)
            k = (255 - 128) / 255
            b = 128
            c_img = c_img.point(lambda x: k * x + b)
        elif task == "depth_pred":
            c_img = c_img
        elif task == "fill":
            c_img = c_img.resize((512, 512)).convert("RGB")
            x1, x2 = fill_x1, fill_x2
            y1, y2 = fill_y1, fill_y2
            mask = Image.new("L", (512, 512), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((x1, y1, x2, y2), fill=255)
            if inpainting:
                mask = Image.eval(mask, lambda a: 255 - a)
            c_img = Image.composite(
                c_img,
                Image.new("RGB", (512, 512), (255, 255, 255)),
                mask
            )
            c_img = Image.composite(
                c_img,
                Image.new("RGB", (512, 512), (128, 128, 128)),
                mask
            )
        elif task == "sr":
            c_img = c_img.resize((int(512 / 4), int(512 / 4))).convert("RGB")
            c_img = c_img.resize((512, 512))

    gen_img = pipe(
        image=c_img,
        prompt=[t_txt.strip()],
        prompt_condition=[c_txt.strip()] if c_txt is not None else None,
        prompt_2=[t_txt],
        height=512,
        width=512,
        num_frames=5,
        num_inference_steps=num_steps,
        guidance_scale=6.0,
        num_videos_per_prompt=1,
        generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed),
        output_type='pt',
        image_embed_interleave=4,
        frame_gap=48,
        mixup=True,
        mixup_num_imgs=2,
        enhance_tp=task in ['subject_driven'],
    ).frames

    output_images = []
    for i in range(10):
        out = gen_img[:, i:i+1, :, :, :]
        out = out.squeeze(0).squeeze(0).cpu().to(torch.float32).numpy()
        out = np.transpose(out, (1, 2, 0))
        out = (out * 255).astype(np.uint8)
        out = Image.fromarray(out)
        output_images.append(out)

    # video = [np.array(img.convert('RGB')) for img in output_images[1:] + [output_images[0]]]
    # video = np.stack(video, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    imageio.mimsave(video_path, output_images[1:]+[output_images[0]], fps=5)

    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    print(f"Peak GPU memory allocated: {peak_memory / 1024**2:.2f} MB")

    return output_images[0], video_path

def get_samples():
    sample_list = [
        {
            "task": "subject_driven", 
            "input": "assets/subject_driven_dreambench_test.jpg", 
            "target_prompt": "a cat in a chef outfit", 
            "condition_image_prompt": "a cat", 
            "output": "assets/subject_driven_dreambench_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "subject_driven", 
            "input": "assets/subject_driven_test.jpg", 
            "target_prompt": "The woman stands in a snowy forest, captured in a half-portrait", 
            "condition_image_prompt": "Woman in cream knit sweater sits calmly by a crackling fireplace, surrounded by warm candlelight and rustic wooden shelves", 
            "output": "assets/subject_driven_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "canny", 
            "input": "assets/canny_test.jpg", 
            "target_prompt": "Mosquito frozen in clear ice cube on sand, glowing sunset casting golden light with misty halo around ice", 
            "condition_image_prompt": "", 
            "output": "assets/canny_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "coloring", 
            "input": "assets/coloring_test.jpg", 
            "target_prompt": "A vibrant young woman with rainbow glasses, yellow eyes, and colorful feather accessory against a bright yellow background", 
            "condition_image_prompt": "", 
            "output": "assets/coloring_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "deblurring", 
            "input": "assets/deblurring_test.jpg", 
            "target_prompt": "Vibrant rainbow ball creates dramatic splash in clear water, bubbles swirling against crisp white background", 
            "condition_image_prompt": "", 
            "output": "assets/deblurring_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "depth", 
            "input": "assets/depth_test.jpg", 
            "target_prompt": "Golden-brown cat-shaped bread loaf with closed eyes rests on wooden table, soft kitchen blur in background", 
            "condition_image_prompt": "", 
            "output": "assets/depth_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "depth_pred", 
            "input": "assets/depth_pred_test.jpg", 
            "target_prompt": "Steaming bowl of ramen with pork slices, soft-boiled egg, greens, and scallions in rich broth on wooden table", 
            "condition_image_prompt": "", 
            "output": "assets/depth_pred_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "fill", 
            "input": "assets/fill_test.jpg", 
            "target_prompt": "Mona Lisa dons a medical mask, her enigmatic smile now concealed beneath crisp white fabric", 
            "condition_image_prompt": "", 
            "output": "assets/fill_ti.png", 
            "inpainting": True,
            "fill_x1": 170,
            "fill_x2": 300,
            "fill_y1": 190,
            "fill_y2": 290,
        },
        {
            "task": "fill", 
            "input": "assets/fill_2_test.jpg", 
            "target_prompt": "Her left hand emerges at the frame's lower right, delicately cradling a vibrant red flower against the black void", 
            "condition_image_prompt": "", 
            "output": "assets/fill_2_ti.png", 
            "inpainting": False,
            "fill_x1": 155,
            "fill_x2": 512,
            "fill_y1": 0,
            "fill_y2": 330,
        },
        {
            "task": "sr", 
            "input": "assets/sr_test.jpg", 
            "target_prompt": "Crispy buffalo wings and golden fries rest on a red-and-white checkered paper lining a gleaming metal tray, with creamy dip", 
            "condition_image_prompt": "", 
            "output": "assets/sr_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
        {
            "task": "style_transfer", 
            "input": "assets/style_transfer_test.jpg", 
            "target_prompt": "bitmoji style. An orange cat sits quietly on the stone slab. Beside it are the green grasses. With its ears perked up, it looks to one side.", 
            "condition_image_prompt": "An orange cat sits quietly on the stone slab. Beside it are the green grasses. With its ears perked up, it looks to one side.", 
            "output": "assets/style_transfer_ti.png", 
            "inpainting": False,
            "fill_x1": None,
            "fill_x2": None,
            "fill_y1": None,
            "fill_y2": None,
        },
    ]

    return [
        [
            sample['task'],
            Image.open(sample['input']),
            sample['target_prompt'],
            sample['condition_image_prompt'],
            Image.open(sample['output']),
            sample['inpainting'],
            sample['fill_x1'],
            sample['fill_x2'],
            sample['fill_y1'],
            sample['fill_y2'],
        ]
        for sample in sample_list
    ]

def create_app():
    with gr.Blocks() as app:
        gr.Markdown(header, elem_id="header")
        gr.Markdown("🚦 To ensure stable model output, we are running the process in a single-threaded serial mode. If your request is queued, please wait patiently for the generation to complete.", elem_id="queue_notice")
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel", elem_classes="inputPanel"):
                condition_image = gr.Image(
                    type="pil", label="Condition Image", width=300, elem_id="input"
                )
                task = gr.Radio(
                    [
                        ("Subject-driven Image Generation", "subject_driven"),
                        ("Canny-to-Image", "canny"),
                        ("Colorization", "coloring"),
                        ("Deblurring", "deblurring"),
                        ("Depth-to-Image", "depth"),
                        ("Depth Prediction", "depth_pred"),
                        ("In/Out-Painting", "fill"),
                        ("Super-Resolution", "sr"),
                        ("Style Transfer", "style_transfer")
                    ],
                    label="Task Selection",
                    value="subject_driven",
                    interactive=True,
                    elem_id="task_selection"
                )
                gr.Markdown(notice, elem_id="notice")
                target_prompt = gr.Textbox(lines=2, label="Target Prompt", elem_id="tp")
                gr.Markdown("**Condition Image Prompt** _(Only required by Subject-driven Image Generation and Style Transfer tasks)_")
                condition_image_prompt = gr.Textbox(lines=2, label="Condition Image Prompt", elem_id="cp")
                random_seed = gr.Number(label="Random Seed", precision=0, value=0, elem_id="seed")
                num_steps = gr.Number(label="Diffusion Inference Steps", precision=0, value=50, elem_id="steps")
                inpainting = gr.Checkbox(label="Inpainting", value=False, elem_id="inpainting")
                fill_x1 = gr.Number(label="In/Out-painting Box Left Boundary", precision=0, value=128, elem_id="fill_x1")
                fill_x2 = gr.Number(label="In/Out-painting Box Right Boundary", precision=0, value=384, elem_id="fill_x2")
                fill_y1 = gr.Number(label="In/Out-painting Box Top Boundary", precision=0, value=128, elem_id="fill_y1")
                fill_y2 = gr.Number(label="In/Out-painting Box Bottom Boundary", precision=0, value=384, elem_id="fill_y2")
                submit_btn = gr.Button("Run", elem_id="submit_btn")

            with gr.Column(variant="panel", elem_classes="outputPanel"):
                # output_image = gr.Image(type="pil", elem_id="output")
                # output_images = gr.Gallery(
                #     label="Output Images",
                #     show_label=True,
                #     elem_id="output_gallery",
                #     columns=1,
                #     rows=10,
                #     object_fit="contain",
                #     height="auto",
                # )
                output_image = gr.Image(
                    type="pil", label="Output Image", elem_id="output_image"
                )
                output_video = gr.Video(
                    label="Output Video", elem_id="output_video"
                )

        with gr.Row():
            examples = gr.Examples(
                examples=get_samples(),
                inputs=[task, condition_image, target_prompt, condition_image_prompt, output_image, inpainting, fill_x1, fill_x2, fill_y1, fill_y2],
                label="Examples",
            )

        submit_btn.click(
            fn=process_image_and_text,
            inputs=[condition_image, target_prompt, condition_image_prompt, task, random_seed, num_steps, inpainting, fill_x1, fill_x2, fill_y1, fill_y2],
            outputs=[output_image, output_video],
        )
        
    return app


if __name__ == "__main__":
    vram_optimization = parse_args()
    init_basemodel(vram_optimization)
    app = create_app()
    app.queue(default_concurrency_limit=1)
    app.launch(debug=True, ssr_mode=False, max_threads=1)