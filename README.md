<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                FS-BAN: Born-Again Networks for <br> Domain Generalization Few-Shot Classification</h1>
<p align='center' style="text-align:center;font-size:1.5em;">
    <a href="https://scholar.google.com/citations?user=kQA0x9UAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Yunqing Zhao</a>&nbsp;,&nbsp;
    <a href="https://sites.google.com/site/mancheung0407/" target="_blank" style="text-decoration: none;">Ngai&#8209;Man Cheung</a></br>
</p>
<p align='center' style="text-align:center;font-size:1.2em;">
<b>Singapore University of Technology and Design (SUTD)</b></br>
<b><em>IEEE Transactions on Image Processing (T-IP), 2023</em></b></br>
</p>

<p align='center';>
<b>
<!-- <em>The Thirty-Sixth Annual Conference on Neural Information Processing Systems (NeurIPS 2022);</em> -->
</b>
</p>

<p align='left' style="text-align:left;font-size:1.32em;">
<b>
    [<a href="https://yunqing-me.github.io/Born-Again-FS/" target="_blank" style="text-decoration: none;">Project Page</a>]&nbsp; /&nbsp;
    [<a href="https://ieeexplore.ieee.org/document/10102807" target="_blank" style="text-decoration: none;">Paper Profile</a>]&nbsp; /&nbsp;
    [<a href="https://drive.google.com/drive/folders/1PIlO7NK8NpwLYUwT76ms_FVca1r0GKkZ?usp=sharing" target="_blank" style="text-decoration: none;">Data Repository</a>]
</b>
</p>


<!-- ---------------------------------------------------------------------- -->

Pytorch implementation for our FS-BAN for cross-domain / domain generalization few-shot classification. With the proposed born-again networks with multi-task learning, we are able to:

1. improve exisiting few-shot classification methods under **cross-domain** setting to stat-of-the-art performance
2. achieve stat-of-the-art performance under **single-domain** few-shot classification setting.

# Installation:

- Platform: Linux
- NVIDIA V100 GPUs with CuDNN 10.1
- PyTorch>=1.4.0
- lmdb, tqdm, wandb

You can install the libiraries through:  `pip install -r requirements.txt`. Alternatively, a suitable conda environment named `fsc` can be created and activated with:

```
conda env create -f environment.yml -n fsc
conda activate fsc
```

Then, clone this repository:
```
git clone https://github.com/yunqing-me/Born-Again-FS.git
cd Born-Again-FS
```

# Datasets
Download 5 datasets seperately with the following commands.
Set `DATASET_NAME` to either: `cars`, `cub`, `miniImagenet`, `places`, or `plantae`.

```
cd filelists
python3 process.py DATASET_NAME
cd ..
```

You may encounter some download issues while processing the datasets, this is due to the original dataset links are invalid. Here, we provide the [data repository](https://drive.google.com/drive/folders/1PIlO7NK8NpwLYUwT76ms_FVca1r0GKkZ?usp=sharing) to help download those datasets. Meanwhile, to download/train/test on `tieredImageNet`, please refer to [Torchmeta](https://github.com/tristandeleu/pytorch-meta).


<!-- # Experiments
## Feature Encoder Pretraining

## Teacher Network Pretraining 

## FS-BAN: Born-Again Networks for DG-FSC

# Evaluation: -->

# Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@article{zhao2023fs,
  title={Fs-ban: Born-again networks for domain generalization few-shot classification},
  author={Zhao, Yunqing and Cheung, Ngai-Man},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

# Acknowledgement

We appreciate the wonderful base implementation of Cross-domain Few-shot Classification from [@Hung-Yu Tseng](https://github.com/hytseng0509/CrossDomainFewShot).

We especially thank for the fruitful discussion with Yiluan Guo (Motional), Jiamei Sun (Microsoft) and Milad Abdollahzadeh (SUTD).



