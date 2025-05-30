# DRA-Ctrl

This is the official implementation of DRA-Ctrl.

## Dimension-Reduction Attack! Video Generative Models are Experts on Controllable Image Synthesis

by *Hengyuan Cao, Yutong Feng, Biao Gong, Yijing Tian, Yunhong Lu, Chuang Liu, and Bin Wang*

[![arXiv](https://img.shields.io/badge/arXiv-2505.23325-b31b1b.svg)](https://arxiv.org/abs/2505.23325)
[![Paper](https://img.shields.io/badge/Paper-PDF-green.svg)](https://arxiv.org/pdf/2505.23325)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://dra-ctrl-2025.github.io/DRA-Ctrl/)
[![HuggingFace](https://img.shields.io/badge/🤗-HF%20Model-yellow)](https://huggingface.co/your-model)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Kunbyte-AI/DRA-Ctrl)

## ✅ TODOs

- [ ] release code
- [ ] release checkpoints

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

![](asset/teaser.png)

## 🚀 Quick Start

### Dependencies
### Checkpoints
### Get Started
```
With cmd
```

```
With Gradio
```

## 📋 Citation

If you find our work helpful, please cite:
```
bibtex
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