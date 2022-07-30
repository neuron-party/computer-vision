from easydict import EasyDict as edict

config = edict()
config.margin_list = [1.0, 0.0, 0.4]
config.embedding_size = 2048
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size=128,
config.lr = 2e-2
config.verbose = 100

# currently unneeded:
config.network = 'r101'
config.optimizer = 'sgd'
config.resume = False
config.output = None