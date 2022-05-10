# load dictionaries with default parameters (from research papers)

def resnet50_params():
    params = {
        'dim':64, 'expansion':4, 'num_classes':1000, 'num_layers': [3, 4, 6, 3], 'layer_dims': [64, 128, 256, 512], 'in_channels':3
    }
    return params

def vit_b16_params(torch=False):
    if torch:
        params = {
            'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'dropout':0.0, 'attention_dropout':0.0,
            'num_layers':12, 'num_heads':8, 'num_classes':1000, 'representation_layer':None, 'fine_tune':None
        }
    else:
        params = {
            'input_channels':3, 'dim':768, 'hidden_dim':3072, 'patch_size':16, 'img_size':224, 'dropout':0.0, 'attention_dropout':0.0,
            'num_layers':12, 'num_heads':8, 'num_classes':1000, 'ft_classes':None, 'ft_classes':None, 'fc_norm':True, 'encoder_norm':True
        }
    return params