# README

Experimental resources for our ACL 2020 Paper "[Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness](https://www.aclweb.org/anthology/2020.acl-main.515/)".

## News

- (2020.7.7) Currently, the provided datasets & code contain involve some unused data, we will remove them and provide the corresponding data format description.
- (2020.7.7) Our code and datasets have been released. We may provide more updates in the future.

## Preparation

Our code is based on the Tensorflow (1.14.0, Python3.6). 

We reuse some codes/scripts from [Tensorflow-NMT](https://github.com/tensorflow/nmt).

### Recommended Environment

```
     conda create -p ~/envs/conkadi python=3.6
     conda activate ~/envs/conkadi 
     conda install tensorflow-gpu==1.14.0
``` 

### Datasets

We use two datasets, and here we provide the processed datasets. If you need the original datasets, please check the cited papers:

- English Reddit([Baidu Disk](https://pan.baidu.com/s/1bHgp0P6oa9szSaVpYDuiMw) (ftdg) / [Google Drive](https://drive.google.com/file/d/1yI9p6w3JRyOrv331Zx-VlF1CGgWuxy_v/view?usp=sharing)):  A subset of [CCM Reddit Commonsense Dataset](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense).

- Chinese Weibo ([Baidu Disk](https://pan.baidu.com/s/1mmAxJ5KzecJVc_MZsnQTuw) (gcfs) / [Google Drive](https://drive.google.com/file/d/10Rzs0afMuP7TQV18EIBp4oXCGV1fGU0i/view?usp=sharing)): It is built upon three previous open Chinese Weibo datasets (please see our paper for detail), and we collected commonsense knowledge facts from [ConceptIO](http://www.conceptnet.io/).

In addition, to evaluate the model, you need to download two pre-trained Embeddings:

- English： [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- Chinese： [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/en/embedding.html)

## Training

We run our experiments on a single Nvidia Geforce RTX2080 Ti with 11GB V-Ram.  We advise you to run our code with at least 11GB V-Ram. We have provided our experiment config files in folder '/config', you may need to the data/model path configurations.


```
mkdir logs
export CUDA_VISIBLE_DEVICES=3
nohup python -u -m KEFU2.run_kefu2 --config=config/weibo >> logs/weibo.txt &
export CUDA_VISIBLE_DEVICES=0
nohup python -u -m KEFU2.run_kefu2 --config=config/reddit >> logs/reddit.txt &
```

## Inference

'-b xx' is used to control the width of beam search.

By default, you can find the generated responses at 'MODEL_PATH/decoded'.

```
export CUDA_VISIBLE_DEVICES=2
python -u -m KEFU2.run_kefu2 --config=config/weibo  --test True -b 1 &
export CUDA_VISIBLE_DEVICES=0
python -u -m KEFU2.run_kefu2 --config=config/reddit  --test True -b 1 &

```

## Evaluation
By default, you can find the results at 'MODEL_PATH/decoded'.

- 'XXXXXX.eres' : entity scores
- 'XXXXXX.res'  : other scores 

```
python -u eval.py --config=config/weibo -b 1 &
python -u eval.py --config=config/reddit -b 1 &


```

### Note
- If you use other evaluation scripts, the results may be different. In our paper, we uniformly evaluate models using the scripts in this project.
- The pre-trained embeddings cannot cover all appeared words, and thus we use random embeddings; therefore, in terms of Embed-AVG/EX, therefore, such results will have minor differences if you repeat multiple evaluations.

# Evaluation Results

Considering the randomness and the difference between the previous code and the released code, we here provide multiple experimental results, for your information. 

| Config | Entity-Match | Entity-Use |  Entity-Recall |
| :-----| :----: | :----: | :----: |
| Weibo@Paper | 1.48 | 2.08 | 0.38 |
| Weibo@Run1 | 1.44 | 2.08 | 0.37 |
| Weibo@Run2 | 1.48 | 2.10 | 0.38 |
| Reddit@Paper | 1.24 | 1.98 | 0.14 |
| Reddit@Run1 | 1.23 | 1.94 | 0.14 |
| Reddit@Run2 | 1.23 | 2.10 | 0.15 |

| Config | Embed-AVG | Embed-EX |  BLEU2 | BLEU3 | Distinct1 | Distinct2 | Entropy |
| :-----| :----: | :----: | :----: | :----: |:----: |:----: |:----: |
| Weibo@Paper | 0.846 | 0.577 | 5.06 | 1.59 |  3.26 |  23.93 |  9.04 |
| Weibo@Run1 | 0.835 | 0.580 | 5.18 | 1.67 |  3.28 |  23.71 |  8.96 |
| Weibo@Run2 | 0.837 | 0.585 | 5.27 | 1.70 |  3.20 |  22.69 |  8.87 |
| Reddit@Paper | 0.867 | 0.852 | 3.53 | 1.27 | 2.77 | 18.78| 8.50 |
| Reddit@Run1 | 0.865 | 0.850 | 4.10 | 1.44 | 2.53| 16.52| 8.33 |
| Reddit@Run2 | 0.863 | 0.852 | 3.70 | 1.29 | 2.80| 19.93| 8.68 |

# Paper Correction

- In Figure3, we wrongly use some early names to refer settings. For correct setting names, please refer to Table 5.

# Citation

If you use our code or data, please kindly cite us in your work.

```
@inproceedings{wu-etal-2020-diverse,
    title = "Diverse and Informative Dialogue Generation with Context-Specific Commonsense Knowledge Awareness",
    author = "Wu, Sixing  and
      Li, Ying  and
      Zhang, Dawei  and
      Zhou, Yang  and
      Wu, Zhonghai",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.515",
    pages = "5811--5820",
    abstract = "Generative dialogue systems tend to produce generic responses, which often leads to boring conversations. For alleviating this issue, Recent studies proposed to retrieve and introduce knowledge facts from knowledge graphs. While this paradigm works to a certain extent, it usually retrieves knowledge facts only based on the entity word itself, without considering the specific dialogue context. Thus, the introduction of the context-irrelevant knowledge facts can impact the quality of generations. To this end, this paper proposes a novel commonsense knowledge-aware dialogue generation model, ConKADI. We design a Felicitous Fact mechanism to help the model focus on the knowledge facts that are highly relevant to the context; furthermore, two techniques, Context-Knowledge Fusion and Flexible Mode Fusion are proposed to facilitate the integration of the knowledge in the ConKADI. We collect and build a large-scale Chinese dataset aligned with the commonsense knowledge for dialogue generation. Extensive evaluations over both an open-released English dataset and our Chinese dataset demonstrate that our approach ConKADI outperforms the state-of-the-art approach CCM, in most experiments.",
}
```