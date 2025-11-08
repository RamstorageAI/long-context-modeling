
### Milestones
"[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](https://arxiv.org/abs/2410.01651)" (ICML 2025) [link](https://github.com/ant-research/long-context-modeling/tree/GCA)

Achieved 1000x extrapolation, but limited by the inability to retrieve every token—only able to retrieve once every S tokens. Random access capability is not flexible enough.

"[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](https://arxiv.org/abs/2504.16795)" (NeurIPS 2025) 

Compared to GCA, token-by-token retrieval has been achieved. But we find its extrapolation ability is not as strong as GCA. We recently found that combining it with a short sliding window instead of Mamba yields stronger extrapolation capability. 

**After the release of this work, we attempted to scale up a larger model and pre-trained it on trillions of tokens. However, we find that its extrapolation capability completely disappeared. Therefore, we strongly recommend using HSA with SWA. We will soon release a tech report on the HSA+SWA-based 8BA1B MoE architecture, which maintains strong extrapolation ability (16M) even after pre-training on trillion of tokens.**

### Core idea of HSA
<img src="figures/hsa_vs_moe.png" width="800">
The core idea of HSA is to perform sparse attention akin to MoE.

Overall, we split a KV cache into fixed-length chunks, each with a summary representation. Each token retrieves top-k chunks via these summary tokens, conducts attention with tokens in each retrieved chunk separately, and then fuses the attention results based on the normalized retrieval scores. 

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