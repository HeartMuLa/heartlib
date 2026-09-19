<p align="center">
    <picture>
        <source srcset="./assets/logo.png" media="(prefers-color-scheme: dark)">
        <img src="./assets/logo.png" width="30%">
    </picture>
    
</p>

<p align="center">
    <a href="https://heartmula.github.io/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="https://arxiv.org/pdf/2601.10547">Paper</a>
    <br>
    <a href="https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year">HeartMuLa-oss-3B-happy-new-year 🤗</a> &nbsp;|&nbsp; <a href="https://modelscope.cn/models/HeartMuLa/HeartMuLa-oss-3B-happy-new-year">HeartMuLa-oss-3B-happy-new-year <picture>
        <source srcset="./assets/badge.svg" media="(prefers-color-scheme: dark)">
        <img src="./assets/badge.svg" width="20px">
    </picture></a>
    <br>
    <a href="https://huggingface.co/HeartMuLa/HeartCodec-oss-encoder">HeartCodec-oss-encoder 🤗</a> &nbsp;|&nbsp; <a href="https://github.com/HeartMuLa/MuLaCover">MuLaCover <picture><source srcset="./assets/github-light.svg" media="(prefers-color-scheme: dark)"><img src="./assets/github.svg" width="20" height="20" alt="GitHub"></picture></a> &nbsp;|&nbsp; <a href="https://huggingface.co/HeartMuLa/MuLaCover">MuLaCover 🤗</a>
</p>

---
# HeartMuLa: A Family of Open Sourced Music Foundation Models

HeartMuLa is a family of open sourced music foundation models including: 
1. HeartMuLa: a music language model that generates music conditioned on lyrics and tags with multilingual support covering almost all languages.
2. HeartCodec: a 12.5 hz music codec with high reconstruction fidelity;
3. HeartTranscriptor: a whisper-based model specifically tuned for lyrics transcription; Check [this page](./examples/README.md) for its usage.
4. HeartCLAP: an audio–text alignment model that establishes a unified embedding space for music descriptions and cross-modal retrieval.
---


Below shows the experiment result of our oss-3B version compared with other baselines.
<p align="center">
    <picture>
        <source srcset="./assets/exp-new.png" media="(prefers-color-scheme: dark)">
        <img src="./assets/exp-new.png" width="90%">
    </picture>
    
</p>

---

## 🔥 Highlight

Our latest internal version of HeartMuLa-7B achieves **comparable performance with Suno** in terms of musicality, fidelity and controllability. 

## 📰 News
Join on Discord! [<img alt="join discord" src="https://img.shields.io/discord/842440537755353128?color=%237289da&logo=discord"/>](https://discord.gg/2Qj5DXsvh)

- 🚀 **16 Sep. 2026**

  Introducing **MuLaCover**, our controllable cover-song and music-remix model!
  Start from reference audio or melody/chord MIDI, keep the original lyrics or
  write new ones, and reshape the genre, instrumentation, and mood through text.
  MuLaCover uses symbolic melody and harmony to preserve the source's musical
  identity while allowing flexible reinterpretation. Code, model weights, and
  generation instructions are now available in the dedicated MuLaCover repository.
  [Code](https://github.com/HeartMuLa/MuLaCover) |
  [Model Weights](https://huggingface.co/HeartMuLa/MuLaCover) |
  [Generation Guide](https://github.com/HeartMuLa/MuLaCover/blob/main/examples/cover_song_generation.md).

- 🚀 **HeartCodec Encoder Release**

  We release **HeartCodec-oss-encoder**, adding audio tokenization and reconstruction
  to HeartCodec. Paired with **HeartCodec-oss-20260123**, it converts mono or stereo
  audio into discrete tokens and reconstructs 48 kHz stereo audio. Encoder and
  decoder checkpoints can be downloaded and loaded separately.

  We hope this contribution will be useful to the music research community.

  [Encoder Weights](https://huggingface.co/HeartMuLa/HeartCodec-oss-encoder) |
  [Reconstruction Guide](./examples/README.md#music-reconstruction).

- 🚀 **10 Apr. 2026**

  We launched online demo spaces on [Hugging Face](https://huggingface.co/spaces/HeartMuLa/heartmula) and [ModelScope](https://www.modelscope.cn/studios/HeartMuLa/heartmula/).

- 🚀 **13 Feb. 2026**

  We released our **HeartMuLa-oss-3B-happy-new-year** version. This version is currently the best open-sourced model in terms of lyrics controllability and music quality. We recommend using **HeartMuLa-oss-3B-happy-new-year** and **HeartCodec-oss-20260123** for music generation.

- ⚖️ **03 Feb. 2026**

  We have released our [HeartMuLa-Benchmark](https://modelscope.cn/datasets/HeartMuLa/HeartMuLa-Benchmark) (referred to as **HeartBeats Benchmark** in our paper) as introduced in our [paper](https://arxiv.org/pdf/2601.10547). This benchmark comprises heterogeneous AI-generated lyrics and tags across diverse languages and genres, providing a rigorous and fair evaluation framework.
  
- 🚀 **23 Jan. 2026**

    By leveraging Reinforcement Learning, we have continuously refined our model and are proud to officially release **HeartMuLa-RL-oss-3B-20260123**. This version is designed to achieve more precise control over styles and tags. Simultaneously, we are launching **HeartCodec-oss-20260123**, which optimizes audio decoding quality.

- 🫶 **20 Jan. 2026** 
    
    [Benji](https://github.com/benjiyaya) has created a wonderful [ComfyUI custom node](https://github.com/benjiyaya/HeartMuLa_ComfyUI) for HeartMuLa. Thanks Benji!

- ⚖️ **20 Jan. 2026** 

    License update: We update the license of this repo and all related model weights to **Apache 2.0**.

- 🚀 **14 Jan. 2026**  
    The official release of **HeartTranscriptor-oss** and the first **HeartMuLa-oss-3B** version along with our **HeartCodec-oss**.

---
## 🧭 TODOs

- ⏳ Release scripts for inference acceleration and streaming inference. The current inference speed is around RTF $\approx 1.0$.
- ⏳ Support **reference audio conditioning**, **fine-grained controllable music generation**, **hot song generation**.
- ⏳ Release the **HeartMuLa-oss-7B** version.
- ✅ Release inference code and pretrained checkpoints of  
  **HeartCodec-oss, HeartMuLa-oss-3B, and HeartTranscriptor-oss**.

---

## 🛠️ Local Deployment

### ⚙️ Environment Setup

We recommend using `python=3.10` for local deployment.

Clone this repo and install locally.

```
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
pip install -e .
```

Download our pretrained checkpoints from huggingface or modelscope using the following command:

```
# if you are using huggingface
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
hf download --local-dir './ckpt/HeartCodec-oss' HeartMuLa/HeartCodec-oss-20260123


# if you are using modelscope
modelscope download --model 'HeartMuLa/HeartMuLaGen' --local_dir './ckpt'
modelscope download --model 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year' --local_dir './ckpt/HeartMuLa-oss-3B'
modelscope download --model 'HeartMuLa/HeartCodec-oss-20260123' --local_dir './ckpt/HeartCodec-oss'

```

After downloading, the `./ckpt` subfolder should structure like this:
```
./ckpt/
├── HeartCodec-oss/
├── HeartMuLa-oss-3B/
├── gen_config.json
└── tokenizer.json
```

For audio reconstruction, download the separate encoder and decoder checkpoints:

```bash
hf download HeartMuLa/HeartCodec-oss-encoder --local-dir ./ckpt/HeartCodec-oss-encoder
hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ./ckpt/HeartCodec-oss-20260123
```

See the [music reconstruction example](./examples/README.md#music-reconstruction) for usage.


### ▶️ Example Usage

To generate music, run:

```
python ./examples/run_music_generation.py --model_path=./ckpt --version="3B"
```

By default this command will generate a piece of music conditioned on lyrics and tags provided in `./assets` folder. The output music will be saved at `./assets/output.mp3`.

#### Use torch.compile for faster generations:

```
python ./examples/run_music_generation.py --model_path=./ckpt --version="3B" --compile --compile_mode max-autotune
```

_Note_: The first run using torch.compile may take longer as it compiles before inference, but once compiled it will not need to re-compile on the next run.
Expected performance improvement using torch.compile is ~2x.
If you are on Windows, you may need to install triton-windows `pip install -U "triton-windows>=3.2,<3.3"`


#### FAQs

1. How to specify lyrics and tags?

    The model will load lyrics from the txt file `--lyrics` link to (by default `./assets/lyrics.txt`). If you would like to use your own lyrics, just modify the content in `./assets/lyrics.txt`. If you would like to save your lyrics to another path, e.g. `my_awesome_lyrics.txt`, remember to input arguments `--lyrics my_awesome_lyrics.txt`.

    For tags it's basically the same.

2. CUDA out of memory?

    If you have multi-GPUs (e.g. 2 4090s), we recommend placing the params of HeartMuLa and HeartCodec separately on different devices. You can do it by typing `--mula_device cuda:0 --codec_device cuda:1`

    If you are running on a single GPU, use `--lazy_load true` so that modules will be loaded on demand and deleted once inference completed to save GPU memory.

All parameters:

- `--model_path` (required): Path to the pretrained model checkpoint
- `--lyrics`: Path to lyrics file (default: `./assets/lyrics.txt`)
- `--tags`: Path to tags file (default: `./assets/tags.txt`)
- `--save_path`: Output audio file path (default: `./assets/output.mp3`)
- `--max_audio_length_ms`: Maximum audio length in milliseconds (default: 240000)
- `--topk`: Top-k sampling parameter for generation (default: 50)
- `--temperature`: Sampling temperature for generation (default: 1.0)
- `--cfg_scale`: Classifier-free guidance scale (default: 1.5)
- `--version`: The version of HeartMuLa, choose between [`3B`, `7B`]. (default: `3B`) # `7B` version not released yet.
- `--mula_device/--codec_device`: The device where params will be placed. Both are set to `cuda` by default. You can use `--mula_device cuda:0 --codec_device cuda:1` to explicitly place different modules to different devices.
- `--mula_dtype/--codec_dtype`: Inference dtype. By default is `bf16` for HeartMuLa and `fp32` for HeartCodec. Setting `bf16` for HeartCodec may result in the degradation of audio quality.
- `--lazy_load`: Whether or not to use lazy loading (default: false). If turned on, modules will be loaded on demand to save GPU usage. 
Recommended format of lyrics and tags:
```txt
[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows,chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy,not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day
```

Regarding tags, check this [issue](https://github.com/HeartMuLa/heartlib/issues/17) for reference.
Our different tags are comma-separated without spaces as illustrated below:
```txt
piano,happy,wedding,synthesizer,romantic
```

---


## ⚖️ License

This repository is licensed under the Apache 2.0 License.

---

## 📚 Citation

```
@misc{yang2026heartmulafamilyopensourced,
      title={HeartMuLa: A Family of Open Sourced Music Foundation Models}, 
      author={Dongchao Yang and Yuxin Xie and Yuguo Yin and Zheyu Wang and Xiaoyu Yi and Gongxi Zhu and Xiaolong Weng and Zihan Xiong and Yingzhe Ma and Dading Cong and Jingliang Liu and Zihang Huang and Jinghan Ru and Rongjie Huang and Haoran Wan and Peixu Wang and Kuoxi Yu and Helin Wang and Liming Liang and Xianwei Zhuang and Yuanyuan Wang and Haohan Guo and Junjie Cao and Zeqian Ju and Songxiang Liu and Yuewen Cao and Heming Weng and Yuexian Zou},
      year={2026},
      eprint={2601.10547},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.10547}, 
}
```

## 📬 Contact
If you are interested in HeartMuLa, feel free to reach us at [contact@mulalabs.ai](mailto:contact@mulalabs.ai).

Welcome to join us through [Discord](https://discord.gg/2Qj5DXsvh) or our WeChat group.

Our WeChat group now has more than 200 members. To join, scan our team member's QR code below and include **HeartMuLa Group Invite** in your friend request. We will invite you into the group manually.
<p align="center">
    <picture>
        <source srcset="./assets/lead_wx.jpeg" media="(prefers-color-scheme: dark)">
        <img src="./assets/lead_wx.jpeg" width="40%">
    </picture>
</p>

### Join MuLa Labs, Vera Praxis Lab

We are always excited to meet people with a strong interest in audio and music. MuLa Labs has internship openings for candidates who want to build the next generation of music and audio technology. To apply, email your resume or CV, together with a short introduction, to [contact@mulalabs.ai](mailto:contact@mulalabs.ai).
