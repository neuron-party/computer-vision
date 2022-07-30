import os
import numbers
import queue as Queue
import threading
import numpy as np
import torch


# used to pass tensors to respective gpus 
class DataLoaderX(torch.utils.data.DataLoader):
    def __init__(self, local_rank, **kwargs):
        super().__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank
        
    def __iter__(self):
        self.iter = super().__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self
    
    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for i in range(len(self.batch)):
                self.batch[i] = self.batch[i].to(device=local_rank, non_blocking=True)
                
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
    
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super().__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()
        
    def run(self):
        torch.cuda.set_device(self.local_rank)
        for i in self.generator:
            self.queue.put(i)
        self.queue.put(None)
        
    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
    
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self