# ResNet

## Network Notes
* **ResNet Head**
     * convolutional layer: project to higher dimension, image downsizing
     * batchnorm, relu: paper specifies it applies nonlinearity after normalization
     * maxpool: downsize image
* **Residual Layer**
    * **Bottleneck**
        * conv1x1: dimension reduction
        * conv3x3: image downsizing
        * conv1x1: dimension restoration
        * downsample (in first bottleneck block of each layer): dimension expansion and downsizing of image when transitioning from one residual layer to the next
    * **Basic**
        * to do
* **MLP Head**
    * adaptive average pooling: resize and flatten output to (dx1) # batch first
    * mlp head: linear layer for classification

## PyTorch Models
`torch_model` and `torch_network`: compatible with torchvision pretrained weights <br>
**Example**
```
import torchvision.models as models
from resnet.torch_model import *
from resnet.torch_network import *

weights = models.resnet50(pretrained=True).state_dict()
params = { ... }
model = ResNet(**params)
model.load_state_dict(weights)
model = model.to(device)
```

## Resources
* https://arxiv.org/pdf/1512.03385.pdf

### todo:
* all resnet variants (bottleneck, basicblock, etc)
* make flexible framework
* document small differences in architecture (i.e pytorch downsizes images in conv3x3 while other networks downsize in first conv1x1)
* add upgrades (zero init, etc)
* clean up code and add better documentation instead of comments everywhere