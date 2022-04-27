# Vision Transformers

## Network Structure
![alt text](https://www.researchgate.net/publication/348947034/figure/fig2/AS:986572736446471@1612228678819/The-Vision-Transformer-architecture-a-the-main-architecture-of-the-model-b-the.png)
`patch-encoding`: divide image into 16x16 non overlapping patches and flatten into 2d matrix
`cls-token`: concatenates a tensor of learnable parameters as input for mlp head
`positional-embedding`: adds learnable positional embeddings to each patch
`multihead-self-attention`: applies attention to the image matrix, calculated by dotting queries and keys, scaling + softmax, and dotting with values
`mlp`: feed forward network
`mlp-head:` linear layer mapping from hidden dimension to number of classes

## PyTorch Models
`torch_model.py` & `torch_network.py`: Vision Transformer implemented using nn.MultiheadAttention, compatible with torchvision.models pretrained weights <br>
**Example**
```
import torchvision.models as models
from vit.torch_network import *
from vit.torch_model import * 

device = torch.device('cuda:0')
weights = models.vit_b_16(pretrained=True)
weights = weights.state_dict()

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':10, 'num_classes':1000}
model = PTVisionTransformer(**params)
model.load_state_dict(weights, strict=False) # strict=False because some weights are unnecessary
```
**Modified MLP Head**
```
model.heads.head = nn.Linear(self.dim, desired_num_classes)
```

## Image Models
`model.py` & `network.py`: Vision Transformer implemented from scratch, compatible with timm pretrained weights (github/rwightman/pytorch-image-models) <br>
**Extra Fine-tuning Layer**
```
import timm
from vit.network import *
from vit.model import *

device = torch.device('cuda:0')
weights = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
weights = weight.state_dict()

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':10, 'num_classes':21843}
model = VisionTransformer(**params)
model.load_state_dict(weights)
model = model.to(device)
```
**Modified MLP Head**
```
import ...

device = torch.device('cuda:0')
weights = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=100)
weights = weights.state_dict()

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':None, 'num_classes':100}
model = VisionTransformer(**params)
model.load_state_dict(state_dict)
model = model.to(device)
```

## Tuning
`to do`: image data augmentations, mixup, randomaug, stochastic depth, augreg, hyperparameter tuning, mask, etc... <br>
* https://arxiv.org/pdf/2106.10270.pdf
* https://arxiv.org/pdf/2112.13492v1.pdf
* https://arxiv.org/pdf/2203.09795.pdf

## Benchmarks:
none yet, still working on this