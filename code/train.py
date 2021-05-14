from importlib import import_module, reload
import torch
import os
import argparse
from torch.utils.data import DataLoader
import wandb
import numpy as np
import random
import json
from tqdm import tqdm



def train(args):
    
    # pytorch version
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))    
    print(f'Device name:{torch.cuda.get_device_name(0)}, The number of usable GPU:{torch.cuda.device_count()}')
    print('=' * 15 + 'training setting' + '=' *15)

    # seed setting
    seed_everything(args.random_seed)

    # device setting
    use_cuda = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장
    
    # -- dataset
    # path of dataset
    train_path = os.path.join(args.dataset_dir, 'train.json')
    val_path = os.path.join(args.dataset_dir, 'val.json')
#     test_path = os.path.join(args.dataset_dir, 'test.json')

    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseDetectionDataset
    train_dataset = dataset_module(
        data_dir=args.dataset_dir,
        annotation=train_path,
    )
    val_dataset = dataset_module(
        data_dir=args.dataset_dir,
        annotation=val_path,
    )
    
#     dataset_module = getattr(import_module("dataset"), args.test_dataset)
#     test_dataset = dataset_module(
#         data_dir=args.dataset_dir,
#         annotation=test_path,
#     )    
    
    # augmentation
    transform_module = getattr(import_module("dataset"), args.train_augmentation)
    train_transform = transform_module()
    train_dataset.set_transform(train_transform)

    transform_module = getattr(import_module("dataset"), args.val_augmentation)
    val_transform = transform_module()
    val_dataset.set_transform(val_transform)

#     transform_module = getattr(import_module("dataset"), args.test_augmentation)
#     test_transform = transform_module()
#     test_dataset.set_transform(test_transform)

    num_classes = train_dataset.num_classes
    
    # -- DataLoader
    train_loader = DataLoader(dataset=train_dataset, 
                               batch_size=args.batch_size,
                               shuffle=True,
                               pin_memory=use_cuda,
                               num_workers=args.num_workers_data_loader,
                               drop_last=True,
                               collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, 
                             batch_size=args.batch_size,
                             shuffle=False,
                             pin_memory=use_cuda,
                             num_workers=args.num_workers_data_loader,
                             drop_last=True,
                             collate_fn=collate_fn)

#     test_loader = DataLoader(dataset=test_dataset,
#                               batch_size=args.batch_size,
#                               pin_memory=use_cuda,
#                               num_workers=args.num_workers_data_loader,
#                               drop_last=True,
#                               collate_fn=collate_fn)

    

    # -- model
    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=num_classes, args=args)
    
    print(f'model: {args.model}')
    # print(model)
#     print(f'model input shape and output shape')
#     x = torch.randn([1, 3, 512, 512])
#     print("input shape : ", x.shape) 
#     out = model(x).to(device)
#     print("output shape : ", out.size())
    
    model = model.to(device)
    
    # -- Loss 
    # loss가 model에 들어가 있는 것 같은데 조절 안 되는건가?
    
    
    # -- Optimizer
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            **args.optimizer_parameter_wo_lr
        )
    
    # -- Scheduler
    if args.scheduler != '':
        schedul_module = getattr(import_module("torch.optim.lr_scheduler"), args.scheduler) # defualt None
        scheduler = schedul_module(
                    optimizer, 
                    **args.scheduler_parameter
                    )
    else:
        # scheduler which do not anything when scheduler.step()
        schedul_module = getattr(import_module("torch.optim.lr_scheduler"),'LambdaLR')
        scheduler = schedul_module(optimizer, lr_lambda=[lambda x: 1])    
    
    print('=' * 15 + 'training started' + '=' *15)
    train_fn(args.epochs, train_loader, val_loader, optimizer, scheduler, model, device)
    print('=' * 15 + 'train ended' + '=' *15)

    
    # save inference config file from the results of training
    with open(args.saved_inference_config_path, 'w') as f_json:
        json.dump(vars(args), f_json)

    print('=' * 7 + f'saved inference configuration on {args.saved_inference_config_path}' + '=' *7)


def train_fn(num_epochs, train_data_loader, val_data_loader, optimizer, scheduler, model, device):
    best_val_loss = np.inf
    loss_hist = Averager()
    loss_history_dict = {}
    for epoch in range(num_epochs):
        # reset loss history
        loss_hist.reset()
        for key in loss_history_dict:
            loss_history_dict[key] = 0

        model.train()
        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # loss history
            _loss_value = 0
            for loss_type_key, loss_value in loss_dict.items():
                loss_history_dict[loss_type_key] = loss_history_dict.get(loss_type_key, 0) + loss_value.item()
                _loss_value += loss_value.item()
            loss_hist.send(_loss_value)
            
            
            # train monitoring in wandb
            if args.is_wandb:  
                wandb.log({'learning rate': optimizer.param_groups[0]["lr"]}, commit=False)
                wandb.log({'train/loss':_loss_value}, commit=False)
                wandb_train_loss_history_dict = {}
                for k, v in loss_dict.items():
                    wandb_train_loss_history_dict['train/' + k] = v.item()
                wandb.log(wandb_train_loss_history_dict)
            
            scheduler.step() # scheduler step per an epoch
            
            
        # train loss report
        print(f'Epoch #{epoch+1} train/loss: {loss_hist.value:.4}')
        for k, v in loss_history_dict.items():
            print(f'train/{k}: {v:.4}', end=', ')
        print()
            
        if (epoch + 1) % args.val_every == 0:
            # validation
            val_loss, val_loss_history_dict = validation(epoch + 1, model, val_data_loader, device)
            if args.is_wandb:
                wandb.log({'val/loss':val_loss}, commit=False)
                wandb_val_loss_history_dict = {}
                for k, v in loss_history_dict.items():
                    wandb_val_loss_history_dict['val/' + k] = v
                wandb.log(wandb_val_loss_history_dict)                
                
            
            # val loss report
            print(f"Epoch #{epoch+1} val/loss: {val_loss}")
            for k, v in val_loss_history_dict.items():
                print(f'val/{k}: {v:.4}', end=', ')
            print()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                args.best_epoch = epoch + 1
                args.best_loss = val_loss
                save_model(model, args.saved_dir, f'{args.saved_model_name}.pth')
            
        

@torch.no_grad()
def validation(epoch, model, data_loader, device):
    print(f'Start validation #{epoch}')
    
    val_loss_history_dict = {}

    cpu_device = torch.device("cpu")
    # coco = get_coco_api_from_dataset(data_loader.dataset)
    
    val_score = 0
    # it should be modified to model.eval()
    model.train()
    # model.eval()
    for images, targets, image_ids in tqdm(data_loader):
        # gpu 계산을 위해 image.to(device)
        
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # calculate loss
        # output = model(images)
        # print(output)
        loss_dict = model(images, targets)
        
        _loss_value = 0
        for loss_type_key, loss_value in loss_dict.items():
            val_loss_history_dict[loss_type_key] = val_loss_history_dict.get(loss_type_key, 0) + loss_value.item()
            _loss_value += loss_value.item()
            

        

        
        # scores = 0
        # for i in range(len(output)):
        #     scores += sum(score for score in output[i]['scores'])
        #
        # val_score += scores
        
    # if type(val_score) == torch.Tensor:
    #     val_score = val_score.item()
    # return val_score
    total_loss = _loss_value
    return total_loss, val_loss_history_dict
            
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


def save_model(model, saved_dir, file_name="model.pth"):
    
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)
    # check_point = {'net': model.state_dict()}
    saved_model_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), saved_model_path)
    print(f'{file_name} saved at {saved_model_path}')
    


# seed 고정
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

    
    

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
    
    # add variable to argments for record
    d = vars(args)
    d['best_epoch'] = -1
    d['best_loss'] = -1
    print('=' * 15 + 'arguments' + '=' *15)  
    print(args)

    if args.is_wandb:
        if args.wandb_experiment_name == "":
            args.wandb_experiment_name = f'{args.model},backbone:{args.backbone},loss:{args.criterion},optm:{args.optimizer},sche:{args.scheduler},bs:{args.batch_size},ep:{args.epochs}'
        wandb.init(project=args.wandb_project_name,
                  group=args.wandb_group,
                  name=args.wandb_experiment_name
                  )
        
    
    train(args)