
### Milestones
"[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](https://arxiv.org/abs/2410.01651)" (ICML 2025) [link](https://github.com/ant-research/long-context-modeling/tree/GCA)

Achieved 1000x extrapolation, but limited by the inability to retrieve every token—only able to retrieve once every S tokens. Random access capability is not flexible enough.

"[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](https://arxiv.org/abs/2504.16795)" (NeurIPS 2025) 

Compared to GCA, token-by-token retrieval has been achieved. But we find its extrapolation ability is not as strong as GCA. We recently found that combining it with a short sliding window instead of Mamba yields stronger extrapolation capability. 

**After the release of this work, we attempted to scale up a larger model and pre-trained it on trillions of tokens. However, we find that the extrapolation capability of Mamba+HSA completely disappeared. Therefore, we strongly recommend using HSA with SWA. We will soon release a tech report on the HSA+SWA-based 8BA1B MoE architecture, which maintains strong extrapolation ability (16M) even after pre-training on trillion of tokens.**



The latest update:

"[Every Token Counts: Generalizing 16M Ultra-Long Context in Large Language Models](https://www.arxiv.org/pdf/2511.23319)"

We **scaled up** our **SWA+HSA architecture** and ran evaluations on several **benchmarks** including **RULER**. By increasing the **SWA window** to $4\text{k}$, the **in-domain** performance was able to roughly **match the baseline**. It is also able to **extrapolate** up to $16\text{M}$ on RULER. However, we observed a **decline** in HSA's **extrapolation capability** as the SWA window grew, *unless* a longer context was utilized. The reasons for this phenomenon are discussed comprehensively in our **technical report**.



### Core idea of HSA

<img src="figures/hsa_vs_moe.png" width="800">



### Results 

![image-20251202120932014](figures\RULER_results)

![image-20251202121057620](figures\benchmark)

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
If you encounter any problems, please feel free to contact us: imhuim982 AT 126.com