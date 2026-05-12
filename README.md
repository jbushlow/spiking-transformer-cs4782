# SpikeGPT Reimplementation

Reimplementation of **SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks** for the Cornell CS 4782 final project. This repository studies whether a spiking language model can preserve useful language understanding and generation behavior while sharply reducing the energy cost of standard transformer-style computation.

## Introduction

Large language models are powerful, but their dense matrix multiplications make training and inference expensive. SpikeGPT replaces standard transformer components with spiking neural network mechanisms built around Leaky Integrate-and-Fire neurons, trading multiply-accumulate-heavy computation for sparse spike-driven updates that are far more energy-efficient.

## Chosen Result

We reimplemented the paper's **46M-parameter SpikeGPT configuration** and trained it on the **Enwik8** byte-level language modeling task. We then evaluated the resulting model on both:

- **Natural Language Generation (NLG)** with Bits per Character (BPC)
- **Natural Language Understanding (NLU)** with downstream classification accuracy

Our goal was to test the paper's central claim: that a spiking language model can remain competitive while dramatically lowering projected energy use.

## GitHub Contents

- [`code/`](./code) contains the model, training scripts, generation scripts, fine-tuning code, and analysis utilities
- [`data/`](./data) contains dataset notes and prepared dataset folders such as `enwik8_split`
- [`results/`](./results) stores checkpoints, tables, beta experiments, and visualization artifacts
- [`poster/`](./poster) contains the course poster for the project
- [`colab_train.ipynb`](./colab_train.ipynb) provides a notebook-based training workflow

## Re-implementation Details

The architecture follows the report's core pipeline:

- **Binary embedding layer** to map bytes into spike-friendly representations
- **12 stacked spiking blocks** with `n_embd = 512`
- **SpikingRWKV** in place of quadratic self-attention
- **Spiking RFFN** in place of a standard transformer feed-forward network
- **Leaky Integrate-and-Fire (LIF) neurons** implemented with SpikingJelly

This design keeps the model recurrent and event-driven, which is where the efficiency gains of SpikeGPT come from.

Pretraining uses **Enwik8** with a next-byte prediction objective. The training stack uses **PyTorch**, **SpikingJelly**, **Adam**, cosine learning-rate decay, and gradient clipping. For NLU evaluation, the project includes fine-tuning and analysis workflows for **SST-2**, **SST-5**, **MR**, and **Subj**.

We also explored one extension beyond the original paper: making the LIF leak parameter **beta** learnable instead of fixed at `0.5`.

## Reproduction Steps

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the 46M SpikeGPT model on Enwik8:

```bash
python code/train.py
```

Resume training from the latest checkpoint:

```bash
python code/train.py --resume latest
```

Generate text from a saved checkpoint:

```bash
python code/generate.py --checkpoint results/checkpoints/epoch_202.pt --prompt "The future of spiking models is"
```

Run the learnable-beta training variant:

```bash
python code/train_learnable_beta.py
```

Fine-tune on the ASCII art workflow included in this repo:

```bash
python code/finetune_ascii.py --pretrained results/checkpoints/epoch_202.pt
```

The full 46M setup is GPU-oriented. Smaller or notebook-friendly experiments can be adapted from the scripts in [`code/config.py`](./code/config.py).

**Google Colab GPU workflow**

This repo also includes [`colab_train.ipynb`](./colab_train.ipynb) for running training on a Google Colab GPU. In Colab, switch the runtime to **GPU**, open the notebook, and run the cells in order. The notebook is set up to mount Google Drive, copy the repo into `/content`, install dependencies, prepare the `enwik8` split, and then launch training from saved checkpoints or from scratch.

**What the main Colab notebook commands do**

- `drive.mount('/content/drive')` connects Colab to Google Drive so checkpoints can persist after the runtime shuts down.
- `cp -r ... /content/` copies the project from Drive into Colab's faster local workspace before training.
- `pip install -r requirements.txt` installs the Python dependencies used by the repo.
- `wget http://mattmahoney.net/dc/enwik8.zip` and `unzip -o enwik8.zip` download the raw Enwik8 dataset.
- The dataset prep cell splits `enwik8` into `train.txt`, `valid.txt`, and `test.txt` inside `data/enwik8_split/`.
- The module refresh cell re-imports local project files after code changes so you do not need to restart the notebook every time.
- The smoke-test cell builds a small SpikeGPT model and runs one forward pass to confirm the dataset, model, and GPU setup are working.
- `python3 code/train.py --resume latest ...` resumes the main 46M training run from the newest checkpoint on Drive.
- `python3 code/train.py --resume none ...` starts a fresh pretraining run.
- `python3 code/train_learnable_beta.py ...` runs the learnable-`beta` extension experiments, either from a chosen checkpoint or from scratch.

## Results/Insights

Our reproduction broadly matched the paper on several NLU benchmarks and preserved the main qualitative result: **SpikeGPT offers a strong energy-efficiency tradeoff even when its language modeling quality trails dense GPT-style models**.

Key takeaways from the report:

- NLU results were generally close to the paper except for a larger gap on **MR**
- NLG results were weaker than the paper, with higher **BPC**
- The architecture was estimated to use about **36.2x less energy** than a standard GPT baseline
- A **learnable beta** showed more promise when introduced earlier in training

## Conclusion

This project reinforced the main lesson of the paper: spiking language models do not yet outperform conventional dense transformers, but they open a compelling path toward lower-power sequence modeling. The drop in performance is real, yet the efficiency gains make SpikeGPT a serious direction for embedded and resource-constrained AI systems.

## References

```bibtex
@article{zhu2023spikegpt,
  title={SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
  author={Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

- Zhu, R. et al. *SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks*. TMLR, 2023. https://arxiv.org/abs/2302.13939
- Cornell CS 4782 Deep Learning course materials

## Acknowledgements

SpikeGPT Reimplementation Final Report: Julian Bushlow, Kathy Chen, John Palsberg, and Angelina Zhou.
