# spiking-transformer-cs4782
Spiking transformer, a reimplementation of SpikeGPT (https://arxiv.org/abs/2302.13939) submitted as a final project for Cornell CS 4782 Spring 2026

FINISH THIS README LATER

Instructions: 

The purpose of the README is to provide a TLDR snapshot of your work targeted towards anyone landing
on your GitHub Repo. This is NOT A report! You can use the sample README here for reference.
Note that most of the content can be copied over concisely from your final report. Limit each section to 1-2
lines/figures. Your README should include the following sections:
1. Introduction
This repository contains our reimplementation of SpikeGPT , a spiking neural network (SNN) language model designed to reduce the energy cost of transformer-based LLMs.

SpikeGPT replaces expensive transformer self-attention operations with spiking recurrent mechanisms inspired by biological neurons, enabling significantly lower energy consumption while maintaining competitive language modeling performance.
2. Chosen Result
We reproduced the 46M parameter SpikeGPT model trained on the Enwik8 dataset and evaluated it on both:

- Natural Language Generation (NLG)
- Natural Language Understanding (NLU)

Our primary goal was reproducing the paper’s claim that SpikeGPT can achieve competitive language modeling performance while drastically reducing energy usage compared to standard GPT architectures.


- Include the relevant figure, table, or equation reference from the original paper.
3. GitHub Contents
- Make a brief note about the content structure of your project.
4. Re-implementation Details
We implemented the SpikeGPT architecture using:

- PyTorch
- SpikingJelly
- Leaky Integrate-and-Fire (LIF) neurons
- RWKV-style recurrent attention replacement

Key architectural details:
- 12 spiking blocks
- 512 embedding dimension
- ~46M parameters
- Binary embedding layer with arctangent surrogate gradients
  
- Include key details about models, datasets, tools, and evaluation metrics.
- Mention any challenges or modifications made to the original approach.
5. Reproduction Steps As meta as this section is, it essentially documents steps someone would need to
follow to implement your GitHub repo in a local environment.
- Describe ”how someone using your GitHub can re-implement your re-implementation?”
- Provide instructions for running your code, including any dependencies, required libraries, and
command-line arguments.
- Specify the computational resources (e.g., GPU) needed to reproduce your results.
6. Results/Insights
  
Our reimplementation achieved results comparable to the original paper on most NLU benchmarks while maintaining substantial projected energy savings.

Key findings:
- Competitive NLU performance relative to the SpikeGPT paper
- Worse BPC than the original implementation, likely due to limited compute budget
- Estimated ~36.2× lower energy usage than standard GPT-style transformers
- Learnable β showed promising early-stage training improvements

- 
- Present your re-implementation results as a comparison to the original paper’s findings. Describes
”what can someone expect as the end-result of using your GitHub repo?”
7. Conclusion
This project demonstrates that spiking neural networks can scale to language modeling tasks while significantly reducing computational cost. Although performance still trails traditional transformers, SpikeGPT highlights a promising direction for energy-efficient LLMs and embedded AI systems.
8. References

```bibtex
@article{zhu2023spikegpt,
  title={SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
  author={Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

Additional references:
- Cornell CS 4782 Deep Learning course materials

9. Acknowledgements

This project was completed as the final project for Cornell CS 4782: Deep Learning (Spring 2026).

We thank the course staff and the authors of SpikeGPT for making their work publicly available.
