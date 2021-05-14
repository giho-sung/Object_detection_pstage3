import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
from importlib import import_module, reload
import argparse
import json

from pytorch_model_summary import summary as ps

'''
model summary to be continue
'''


if __name__ == '__main__':
    # set parser
    set_parser = getattr(import_module("common"), 'set_parser')
    parser = set_parser()

    # get arguments from parser
    args, unknown = parser.parse_known_args()    
    
    # put arguments from config_file
    print(f'from_only_config: {args.from_only_config}')
    if args.from_only_config:
        # load config file
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        args = argparse.Namespace()
        d = vars(args)
        for key, value in config_dict.items():
            d[key] = value
            
    # basic parameter
    num_classes = 11
    batch_size = args.batch_size
    
    # make random image and targets
    
    images, boxes = torch.rand(batch_size, 3, 512, 512), torch.randint(1, 200, (batch_size, 11, 4))
    labels = torch.randint(0, num_classes, (batch_size, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['boxes'][:,2] = d['boxes'][:,0] + d['boxes'][:,2]
        d['boxes'][:,3] = d['boxes'][:,1] + d['boxes'][:,3]
        d['labels'] = labels[i]
        targets.append(d)    
        
    # -- model
    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=num_classes, args=args)
    
    print(f'model: {args.model}')

    print(f'==pytorch_model_summary==')
    print(ps(model, torch.zeros(1, 3, 512, 512), show_input=True, show_hierarchical=True)) 

    print(f'==pytorch profiling==')
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_train"):
            model(images, targets)
    
    # sort_by : 'cpu_time_total', 'self_cpu_memory_usage',    'cpu_memory_usage',       
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))