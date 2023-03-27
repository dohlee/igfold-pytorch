# igfold-pytorch

![igfold_model](img/banner.png)

Re-implementation of IgFold, a fast antibody structure prediction method, in PyTorch. You can find the official implementation of IgFold [here](https://github.com/Graylab/IgFold/tree/main/igfold).

## Installation
```shell
$ pip install igfold-pytorch
```

## Usage
```python
from igfold_pytorch import IgFold

bsz = 1
x = torch.randn([bsz, 128, 512])        # Embedding vectors from AntiBERTy
e = torch.randn([bsz, 128, 128, 512])   # Attention matrices from AntiBERTy
r = torch.randn([bsz, 128, 512])        # Template backbone rotations
t = torch.randn([bsz, 128, 512])        # Template backbone translations

model = IgFold()
result = model(x, e, r, t) # result['x'], result['e'], result['coords']
```

## Citation
```bibtex
@article{ruffolo2022fast,
    title = {Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies},
    author = {Ruffolo, Jeffrey A and Chu, Lee-Shin and Mahajan, Sai Pooja and Gray, Jeffrey J},
    journal = {bioRxiv},
    year= {2022}
}
```
