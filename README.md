# DRA-Ctrl

This is the official implementation of DRA-Ctrl.

## Dimension-Reduction Attack! Video Generative Models are Experts on Controllable Image Synthesis

by *Hengyuan Cao, Yutong Feng, Biao Gong, Yijing Tian, Yunhong Lu, Chuang Liu, and Bin Wang*

[![arXiv](https://img.shields.io/badge/arXiv-2505.23325-b31b1b.svg)](https://arxiv.org/abs/2505.23325)
[![Paper](https://img.shields.io/badge/Paper-PDF-green.svg)](https://arxiv.org/pdf/2505.23325)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://dra-ctrl-2025.github.io/DRA-Ctrl/)
[![HuggingFace](https://img.shields.io/badge/🤗-HF%20Model-yellow)](https://huggingface.co/Kunbyte/DRA-Ctrl)
[![HuggingFace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces/Kunbyte/DRA-Ctrl)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Kunbyte-AI/DRA-Ctrl)

## Updates
[2025-07-10] When the model is not in use, move it to the CPU to reduce GPU memory usage, and apply quantization to further decrease memory requirements. Now our work should be able to run on consumer-grade GPUs. For specific usage, please refer to the [Get Started](#get-started) section below. Please note:​​ You need to **check the requirements.txt** file to update environment dependencies, as this ensures the new features will function properly.

[2025-07-01] Added a new Gradio app (gradio_app_hf.py) designed similarly to our HuggingFace Space, making it easier to switch tasks, adjust parameters, and directly test examples. The previous Gradio app (gradio_app.py) will remain unchanged.

## ✅ TODOs

- [x] release code
- [x] release checkpoints
- [x] use quantized version to save VRAM
- [ ] using FramePack as base model

## 🔍 Introduction

> **<h3>Abstract</h3>**
> Video generative models can be regarded as world simulators due to their ability to capture dynamic, continuous changes inherent in real-world environments. 
> These models integrate high-dimensional information across visual, temporal, spatial, and causal dimensions, enabling predictions of subjects in various status. 
> A natural and valuable research direction is to explore whether a fully trained video generative model in high-dimensional space can effectively support lower-dimensional tasks such as controllable image generation. 
> In this work, we propose a paradigm for video-to-image knowledge compression and task adaptation, termed *Dimension-Reduction Attack* (`DRA-Ctrl`), which utilizes the strengths of video models, including long-range context modeling and flatten full-attention, to perform various generation tasks. 
> Specially, to address the challenging gap between continuous video frames and discrete image generation, we introduce a mixup-based transition strategy that ensures smooth adaptation. 
> Moreover, we redesign the attention structure with a tailored masking mechanism to better align text prompts with image-level control. 
> Experiments across diverse image generation tasks, such as subject-driven and spatially conditioned generation, show that repurposed video models outperform those trained directly on images. 
> These results highlight the untapped potential of large-scale video generators for broader visual applications. 
> `DRA-Ctrl` provides new insights into reusing resource-intensive video models and lays foundation for future unified generative models across visual modalities.

![](assets/teaser.png)

## 🚀 Quick Start

### Hardware Requirements
Our method is implemented on Linux with H800 80GB GPU.
The peak VRAM consumption stays below 45GB.

### Dependencies
```
conda create --name dra_ctrl python=3.12
pip install -r requirements.txt
```
### Checkpoints
We use [the community fork](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-I2V)  for Diffusers-format weights on [tencent/HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V) as the initialization parameters for the model.

You can download the LoRA weights for various tasks of DRA-Ctrl at [this link](https://huggingface.co/Kunbyte/DRA-Ctrl).

The checkpoint directory is shown below.
```
DRA-Ctrl/
└── ckpts/
    ├── HunyuanVideo-I2V/
    |   ├── image_processor/
    |   ├── scheduler/
    |       ...
    ├── depth-anything-small-hf
    |   ├── model.safetensors
    |       ...
    ├── canny.safetensors
    ├── coloring.safetensors
    ├── deblurring.safetensors
    ├── depth.safetensors
    ├── depth_pred.safetensors
    ├── fill.safetensors
    ├── sr.safetensors
    ├── subject_driven.safetensors
    └── style_transfer.safetensors
```

### Get Started
To reduce GPU memory requirements, we provide a parameter `vram_optimization` to specify different levels of memory optimization schemes. The specific parameters and their meanings are as follows:

`No_Optimization`: No optimization is applied, and **48GB** of VRAM is sufficient to run the code.

`HighRAM_HighVRAM`: No more than **20GB** of VRAM is required.

`HighRAM_LowVRAM`: No more than **8GB** of VRAM is required.

`LowRAM_HighVRAM`: No more than **20GB** of VRAM is required.

`LowRAM_LowVRAM`: No more than **8GB** of VRAM is required.

`VerylowRAM_LowVRAM`: No more than **8GB** of VRAM is required.

**Note**: Reduced resources will lead to increased generation time.

```
python gradio_app_hf.py --vram_optimization SET_YOUR_OPTIMIZATION_SCHEME_HERE
```

Here is the command to run the legacy Gradio app, ​which we **do not recommend using**. For easier switching between tasks, adjusting parameters, testing examples, and better VRAM optimization, please use the command above.

```
python gradio_app.py --config configs/gradio.yaml
```

In ​spatially-aligned image generation tasks, when passing the condition image to `gradio_app`, there's no need to manually input edge maps, depth maps, or other condition images - only the original image is required. The corresponding condition images will be automatically extracted.

You can use the `*_test.jpg` or `*_test.png` images from the assets folder as ​condition images​ for input to `gradio_app`, which will generate the following examples:

Examples:
|               Task              |          Condition Image         |                                                                 Target Prompt                                                                |                                                     Condition Image Prompt                                                    |           Target Image           |
|:-------------------------------:|:--------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:--------------------------------:|
|          Canny to Image         |      ![](assets/canny_ci.png)     |                   Mosquito frozen in clear ice cube on sand, glowing sunset casting golden light with misty halo around ice                  |                                                               -                                                               |      ![](assets/canny_ti.png)     |
|           Colorization          |    ![](assets/coloring_ci.png)    |          A vibrant young woman with rainbow glasses, yellow eyes, and colorful feather accessory against a bright yellow background          |                                                               -                                                               |    ![](assets/coloring_ti.png)    |
|            Deblurring           |   ![](assets/deblurring_ci.png)   |                 Vibrant rainbow ball creates dramatic splash in clear water, bubbles swirling against crisp white background                 |                                                               -                                                               |   ![](assets/deblurring_ti.png)   |
|          Depth to Image         |      ![](assets/depth_ci.png)     |                  Golden-brown cat-shaped bread loaf with closed eyes rests on wooden table, soft kitchen blur in background                  |                                                               -                                                               |      ![](assets/depth_ti.png)     |
|         Depth Prediction        |   ![](assets/depth_pred_ci.png)   |                 Steaming bowl of ramen with pork slices, soft-boiled egg, greens, and scallions in rich broth on wooden table                |                                                               -                                                               |   ![](assets/depth_pred_ti.png)   |
|         In/Out-painting         |      ![](assets/fill_2_ci.png)      |                          Her left hand emerges at the frame's lower right, delicately cradling a vibrant red flower against the black void                         |                                                               -                                                               |      ![](assets/fill_2_ti.png)      |
|         In/Out-painting         |      ![](assets/fill_ci.png)      |                          Mona Lisa dons a medical mask, her enigmatic smile now concealed beneath crisp white fabric                         |                                                               -                                                               |      ![](assets/fill_ti.png)      |
|         Super-resolution        |       ![](assets/sr_ci.png)       |          Crispy buffalo wings and golden fries rest on a red-and-white checkered paper lining a gleaming metal tray, with creamy dip         |                                                               -                                                               |       ![](assets/sr_ti.png)       |
| Subject-driven image generation | ![](assets/subject_driven_ci.jpg) |                                                            The woman stands in a snowy forest, captured in a half-portrait outfit                                                            |                                                             Woman in cream knit sweater sits calmly by a crackling fireplace, surrounded by warm candlelight and rustic wooden shelves                                                             | ![](assets/subject_driven_ti.png) |
| Subject-driven image generation | ![](assets/subject_driven_dreambench_ci.jpg) |                                                            a cat in a chef outfit outfit                                                            |                                                             a cat                                                             | ![](assets/subject_driven_dreambench_ti.png) |
|          Style Transfer         | ![](assets/style_transfer_ci.jpg) | bitmoji style. An orange cat sits quietly on the stone slab. Beside it are the green grasses. With its ears perked up, it looks to one side. | An orange cat sits quietly on the stone slab. Beside it are the green grasses. With its ears perked up, it looks to one side. | ![](assets/style_transfer_ti.png) |

## 📋 Citation

If you find our work helpful, please cite:
```
@misc{cao2025dimensionreductionattackvideogenerative,
      title={Dimension-Reduction Attack! Video Generative Models are Experts on Controllable Image Synthesis}, 
      author={Hengyuan Cao and Yutong Feng and Biao Gong and Yijing Tian and Yunhong Lu and Chuang Liu and Bin Wang},
      year={2025},
      eprint={2505.23325},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23325}, 
      }
```

## Attribution
This project uses code from the following sources:
- [diffusers/models/transformers/transformer_hunyuan_video](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hunyuan_video.py) - Copyright 2024 The HunyuanVideo Team and The HuggingFace Team (Apache 2.0 licensed).
- [diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_image2video](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuan_video/pipeline_hunyuan_video_image2video.py) - Copyright 2024 The HunyuanVideo Team and The HuggingFace Team (Apache 2.0 licensed).

## Acknowledgements
We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co/) repositories, for their open research and exploration.