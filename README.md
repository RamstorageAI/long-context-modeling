
### Milestones
"[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](https://arxiv.org/abs/2410.01651)" (ICML 2025) [link](https://github.com/ant-research/long-context-modeling/tree/GCA)

Achieved 1000x extrapolation, but limited by the inability to retrieve every token—only able to retrieve once every S tokens. Random access capability is not flexible enough.

"[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](https://arxiv.org/abs/2504.16795)" (NeurIPS 2025) (Code will be released soon)

Token-by-token retrieval has been achieved, but its extrapolation ability is not as strong as GCA. We recently found that combining it with a short sliding window instead of Mamba yields stronger extrapolation capability.

### Model Architecture (To be updated for HSA)
<img src="figures/gca_model_arch.png" width="800">
When generating the current chunk (c7), GCA (Grouped CA) retrieves past chunks using the landmark representation of c6 to assist in token prediction for the next chunk. The key to GCA's length generalization lies in an end-to-end differentiable retrieval mechanism, which is achieved through a two-stage attention mechanism. After selecting the top-k chunks:

In the first stage, each token in c7 performs attention with the tokens within the retrieved chunk respectively to obtain information from that chunk. Taking the example in the diagram, $x^{20}$ interacts with the tokens of the i-th retrieved chunk through attention, resulting in the corresponding output $O_{20}^i$.

In the second stage, the softmax-normalized retrieval scores of the chunks are used as weights to perform a weighted summation of $O_{20}^i$, thereby incorporating the retrieval scores into the forward propagation process.

During backpropagation (BP), the weights of past chunks that better facilitate token prediction for the next chunk will be enhanced, enabling end-to-end causal retrieval learning.

<!--The critical aspect is that tokens in c7 perform cross-attention with each retrieved chunk to obtain chunk-level information. Finally, this information is fused using weights derived from a softmax over the retrieval scores, allowing the retrieval scores to participate in the forward process and making it differentiable.
-->

### Results (To be updated for HSA)

<img src="figures/key_results.png" width="800">
All models were pre-trained on contexts of no more than 16K tokens, and all attention spans are limited to no more than 728 tokens. Our model (DRT) achieves 1000x extrapolation on the needle-in-a-haystack task, maintaining high accuracy even with 16M context length.

### Environments
torch==2.4.0, transformers>=4.36.0, triton==3.0.0

`pip install requirements.txt`

### Data Preparation

Before pre-training, ensure that the corpus is indexed. Pre-processing script:

Pile: `python preprocess/pile_neox.py`



### Unittests

Test triton kernel:

`pytest ops/hsa_tritoin.py`


### Pre-training

`sh scripts/pretrain_pile/pretrain_model.sh`


### Contact
If you encounter any problems, please feel free to contact us: aaron.hx AT antgroup.com