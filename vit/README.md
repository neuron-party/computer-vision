# Vision Transformers

## Network Structure
`patch_embedding` :
convert image of size (b, c, h, w) into patches (b, d, ph, pw), flatten into 2d matrix of shape (b, s, d) <br>
* s : sequence length = (h / ph) * (w / pw)
* d : dimension = out_channels in convolutional layer <br>

`cls_token` :
concatenates a learnable nn.Parameter() tensor of size (1, s, d) to input matrix, gathers information of patch embeddings when fed through the Encoder
* s = s + 1 <br>

`pos_embedding` :
add optional learnable nn.Parameter() tensor of size (1, s, d) to input matrix <br>

`encoder` :
layernorm -> multiheaded self attention -> layernorm -> feed forward mlp <br>
* MHSA 
    * linear projection of input matrix -> q k v
    * query @ key -> scores @ values -> attention-applied output
    * intuition: network learns the optimal projection weights in applying more/less attention to certain patches in differentiating classes <br>

`mlp-head` :
takes attention-encoded cls token and outputs classification predictions <br>

`fine-tuning` :
additional linear layer with desired num_classes as output dimension

## PyTorch Models
`torch_model.py` & `torch_network.py`: Vision Transformer implemented using nn.MultiheadAttention, compatible with torchvision.models pretrained weights <br>
**Example**
```
import torchvision.models as models
from vit.torch_network import *
from vit.torch_model import * 

device = torch.device('cuda:0')
weights = vit_b_16(pretrained=True)
weights = weights.to(device)
torch.save(weights.state_dict(), PATH)

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':10, 'num_classes':21843}
model = PTVisionTransformer(**params)
model.load_state_dict(torch.load(PATH, map_location=device), strict=False) # strict=False because some weights are unnecessary
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
weights = weights.to(device)
torch.save(weights.state_dict(), PATH)

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':10, 'num_classes':21843}
model = VisionTransformer(**params)
model.load_state_dict(torch.load(PATH, map_location=device), strict=False) # strict=False because some weights are unnecessary
```
**Modified MLP Head**
```
import ...

device = torch.device('cuda:0')
weights = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=100)
weights = weights.to(device)
torch.save(weights.state_dict(), PATH)

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':None, 'num_classes':100}
model = VisionTransformer(**params)
model.load_state_dict(torch.load(PATH, map_location=device), strict=False) # strict=False because some weights are unnecessary
```
## Benchmarks:
none yet, still working on this