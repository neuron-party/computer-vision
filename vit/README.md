# Vision Transformers

## PyTorch Models
`torch_model.py` & `torch_network.py`: Vision Transformer implemented using nn.MultiheadAttention, compatible with torchvision.models pretrained weights

## Image Models
`model.py` & `network.py`: Vision Transformer implemented from scratch, compatible with timm pretrained weights (github/rwightman/pytorch-image-models) 
**example**
```
import timm
from vit.network import *
from vit.model import *

device = torch.device('cuda:0')
weights = timm.create_model('vit_base_patch16_224_in21k')
weights = weights.to(device)
torch.save(weights.state_dict(), PATH)

params = {'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'num_layers':12, 
          'dropout':0.0, 'attention_dropout':0.0, 'num_heads':12, 'fine_tune':10, 'num_classes':21843}
model = VisionTransformer(**params)
model.load_state_dict(torch.load(PATH, map_location=device), strict=False) # strict=False because some weights are unnecessary
```