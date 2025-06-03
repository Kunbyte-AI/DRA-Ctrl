import os
import sys
import torch
import diffusers
import transformers
import argparse
import peft
import copy
import cv2
import gradio as gr
import numpy as np

from peft import LoraConfig
from omegaconf import OmegaConf
from safetensors.torch import safe_open
from PIL import Image, ImageDraw, ImageFilter

from models import HunyuanVideoTransformer3DModel
from pipelines import HunyuanVideoImageToVideoPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="DRA-Ctrl Gradio App")
    parser.add_argument("--config", type=str, default=None, required=True, help="path to config")
    args = parser.parse_args()
    return args.config

def init_pipeline(args):
    global pipe
    
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(f'{args.i2v_model_root}/transformer', inference_subject_driven=args.task in ['subject_driven'])
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
    vae = diffusers.AutoencoderKLHunyuanVideo.from_pretrained(f'{args.i2v_model_root}/vae')
    text_encoder = transformers.LlavaForConditionalGeneration.from_pretrained(f'{args.i2v_model_root}/text_encoder')
    text_encoder_2 = transformers.CLIPTextModel.from_pretrained(f'{args.i2v_model_root}/text_encoder_2')
    tokenizer = transformers.AutoTokenizer.from_pretrained(f'{args.i2v_model_root}/tokenizer')
    tokenizer_2 = transformers.CLIPTokenizer.from_pretrained(f'{args.i2v_model_root}/tokenizer_2')
    image_processor = transformers.CLIPImageProcessor.from_pretrained(f'{args.i2v_model_root}/image_processor')

    device = 'cuda:0'
    weight_dtype = torch.bfloat16

    transformer.requires_grad_(False)
    vae.requires_grad_(False).to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False).to(device, dtype=weight_dtype)
    text_encoder_2.requires_grad_(False).to(device, dtype=weight_dtype)
    transformer.to(device, dtype=weight_dtype)
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
        lora_forward = module.forward
        non_lora_forward = module.base_layer.forward
        img_sequence_length = int((args.img_size / 8 / 2) ** 2)
        encoder_sequence_length = 144 + 252 # encoder sequence: 144 img 252 txt
        num_imgs = 4
        num_generated_imgs = 3
        num_encoder_sequences = 2 if args.task in ['subject_driven', 'style_transfer'] else 1

        def hacked_lora_forward(self, x, *args, **kwargs):
            if x.shape[1] == img_sequence_length * num_imgs and len(x.shape) > 2:
                return torch.cat((
                    lora_forward(x[:, :-img_sequence_length*num_generated_imgs], *args, **kwargs),
                    non_lora_forward(x[:, -img_sequence_length*num_generated_imgs:], *args, **kwargs)
                ), dim=1)
            elif x.shape[1] == encoder_sequence_length * num_encoder_sequences or x.shape[1] == encoder_sequence_length:
                return lora_forward(x, *args, **kwargs)
            elif x.shape[1] == img_sequence_length * num_imgs + encoder_sequence_length * num_encoder_sequences:
                return torch.cat((
                    lora_forward(x[:, :(num_imgs - num_generated_imgs)*img_sequence_length], *args, **kwargs),
                    non_lora_forward(x[:, (num_imgs - num_generated_imgs)*img_sequence_length:-num_encoder_sequences*encoder_sequence_length], *args, **kwargs),
                    lora_forward(x[:, -num_encoder_sequences*encoder_sequence_length:], *args, **kwargs)
                ), dim=1)
            elif x.shape[1] == 3072:
                return non_lora_forward(x, *args, **kwargs)
            else:
                raise ValueError(
                    f"hacked_lora_forward receives unexpected sequence length: {x.shape[1]}, input shape: {x.shape}!"
                )

        return hacked_lora_forward.__get__(module, type(module))

    for n, m in transformer.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)

    if args.task == 'canny':
        model_root = args.canny_model_root
    elif args.task == 'coloring':
        model_root = args.coloring_model_root
    elif args.task == 'deblurring':
        model_root = args.deblurring_model_root
    elif args.task == 'depth':
        model_root = args.depth_model_root
    elif args.task == 'depth_pred':
        model_root = args.depth_pred_model_root
    elif args.task == 'fill':
        model_root = args.fill_model_root
    elif args.task == 'sr':
        model_root = args.sr_model_root
    elif args.task == 'subject_driven':
        model_root = args.subject_driven_model_root
    elif args.task == 'style_transfer':
        model_root = args.style_transfer_model_root
    else:
        raise ValueError(f"Unknown task: {args.task}")

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


def process_image_and_txt(c_img, t_txt, c_txt, args):
    c_txt = None if c_txt == "" else c_txt

    # resize image
    c_img = c_img.resize((512, 512))

    save_dir = os.path.join(args.log_dir, args.task, f"{t_txt.replace(' ', '_')[:30]}_seed_{args.random_seed}")
    os.makedirs(save_dir, exist_ok=True)

    if args.task not in ['subject_driven', 'style_transfer']:
        if args.task == "canny":
            def get_canny_edge(img):
                img_np = np.array(img)
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
                edges_tmp = Image.fromarray(edges).convert("RGB")
                edges_tmp.save(os.path.join(save_dir, f"edges.png"))
                edges[edges == 0] = 128
                return Image.fromarray(edges).convert("RGB")
            c_img = get_canny_edge(c_img)
        elif args.task == "coloring":
            c_img = (
                c_img.resize((args.img_size, args.img_size))
                .convert("L")
                .convert("RGB")
            )
        elif args.task == "deblurring":
            blur_radius = 10
            c_img = (
                c_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .resize((args.img_size, args.img_size))
                .convert("RGB")
            )
        elif args.task == "depth":
            def get_depth_map(img):
                from transformers import pipeline

                pipe = pipeline(
                    task="depth-estimation",
                    model=args.depth_anything_model_root,
                    device="cpu",
                )
                return pipe(img)["depth"].convert("RGB").resize((args.img_size, args.img_size))
            c_img = get_depth_map(c_img)
            c_img.save(os.path.join(save_dir, f"depth.png"))
            k = (255 - 128) / 255
            b = 128
            c_img = c_img.point(lambda x: k * x + b)
        elif args.task == "depth_pred":
            c_img = c_img
        elif args.task == "fill":
            c_img = c_img.resize((args.img_size, args.img_size)).convert("RGB")
            x1, x2 = args.fill_x1, args.fill_x2
            y1, y2 = args.fill_y1, args.fill_y2
            mask = Image.new("L", (args.img_size, args.img_size), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((x1, y1, x2, y2), fill=255)
            if args.inpainting:
                mask = Image.eval(mask, lambda a: 255 - a)
            c_img = Image.composite(
                c_img,
                Image.new("RGB", (args.img_size, args.img_size), (255, 255, 255)),
                mask
            )
            c_img.save(os.path.join(save_dir, f"mask.png"))
            c_img = Image.composite(
                c_img,
                Image.new("RGB", (args.img_size, args.img_size), (128, 128, 128)),
                mask
            )
        elif args.task == "sr":
            c_img = c_img.resize((int(args.img_size / 4), int(args.img_size / 4))).convert("RGB")
            c_img.save(os.path.join(save_dir, f"low_resolution.png"))
            c_img = c_img.resize((args.img_size, args.img_size))
            c_img.save(os.path.join(save_dir, f"low_to_high.png"))

    if pipe is None:
        init_pipeline(args)

    gen_img = pipe(
        image=c_img,
        prompt=[t_txt.strip()],
        prompt_condition=[c_txt.strip()] if c_txt is not None else None,
        prompt_2=[t_txt],
        height=args.img_size,
        width=args.img_size,
        num_frames=5,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=1,
        generator=torch.Generator(device=pipe.transformer.device).manual_seed(args.random_seed),
        output_type='pt',
        image_embed_interleave=4,
        frame_gap=48,
        mixup=True,
        mixup_num_imgs=2,
    ).frames

    # save all generated images
    for i in range(gen_img.shape[1]):
        gen_img_i = gen_img[:, i:i+1, :, :, :]
        gen_img_i = gen_img_i.squeeze(0).squeeze(0).cpu().to(torch.float32).numpy()
        gen_img_i = np.transpose(gen_img_i, (1, 2, 0))
        gen_img_i = (gen_img_i * 255).astype(np.uint8)
        gen_img_i = Image.fromarray(gen_img_i)
        gen_img_i.save(os.path.join(save_dir, f"gen_{i}.png"))

    gen_img = gen_img[:, 0:1, :, :, :]
    gen_img = gen_img.squeeze(0).squeeze(0).cpu().to(torch.float32).numpy()
    gen_img = np.transpose(gen_img, (1, 2, 0))
    gen_img = (gen_img * 255).astype(np.uint8)
    gen_img = Image.fromarray(gen_img)

    return gen_img

def main():
    args = OmegaConf.load(parse_args())
    init_pipeline(args)

    demo = gr.Interface(
    fn=lambda c_img, t_txt, c_txt: process_image_and_txt(c_img, t_txt, c_txt, args),
        inputs=[
            gr.Image(type="pil"),
            gr.Textbox(lines=2),
            gr.Textbox(lines=2),
        ],
        outputs=gr.Image(type="pil"),
        title="DRA-Ctrl Gradio App",
    )

    try:
        demo.launch()
    except Exception as e:
        print("Lauch failed:", e)

if __name__ == "__main__":
    main()