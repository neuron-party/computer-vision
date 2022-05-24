# convnext 

## note:
For some reason torchvision's convnext model replaces both conv1x1 layers with linear layers + input permuting. I will implement 2 architectures: one that matches the design that is covered in the official paper, and one that enables 1-1 weight loading with torchvision's models. 