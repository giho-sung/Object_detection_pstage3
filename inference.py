import argparse
import json
import torch
import albumentations as A
from torch.utils.data import DataLoader
from pycocotools.coco import COCO


from importlib import import_module, reload
import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm




def inference_fn(data_loader, model, device):
    outputs = []
    for images, targets, image_ids in tqdm(data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.float().to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs



def inference(args):
    # pytorch version
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))    
    print(f'Device name:{torch.cuda.get_device_name(0)}, The number of usable GPU:{torch.cuda.device_count()}')

    # device setting
    use_cuda = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU 사용 가능 여부에 따라 device 정보 저장
    
    print('=' * 15 + 'Inference setting' + '=' *15)  
    
    
    # -- Dataset
    test_path = os.path.join(args.dataset_dir, 'test.json')

    dataset_module = getattr(import_module("dataset"), args.dataset) # default: BaseDetectionDataset
    test_dataset = dataset_module(
        data_dir=args.dataset_dir,
        annotation=test_path,
    )    
    
    # -- Augmentation
    transform_module = getattr(import_module("dataset"), args.test_augmentation) # TestAugmentation
    test_transform = transform_module()
    test_dataset.set_transform(test_transform)

    def collate_fn(batch):
        return tuple(zip(*batch))
    
    num_classes = test_dataset.num_classes
    
    # -- DataLoader
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers_data_loader,
        collate_fn=collate_fn
    )
    
    # -- model
    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=num_classes, args=args)
    model = model.to(device)
    # load model weights
    saved_model_path = os.path.join(args.saved_dir, f'{args.saved_model_name}.pth')
    model.load_state_dict(torch.load(saved_model_path))
    
    # -- inference
    model.eval()
    outputs = inference_fn(test_data_loader, model, device)
    
    # -- save submission file
    if not os.path.isdir(args.submission_dir):
        os.mkdir(args.submission_dir)
    
    prediction_strings = []
    file_names = []
    coco = COCO(test_path)
    score_threshold = 0.05
    
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold:
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    
    submssion_path = os.path.join(args.submission_dir, f'{args.saved_model_name}.csv')
    submission.to_csv(submssion_path, index=None)
    print(f'submission file: {args.saved_model_name}.csv saved in {args.submission_dir}')
    print(submission.head())

    
    

if __name__ == '__main__':
    
    # set parser
    set_parser = getattr(import_module("common"), 'set_parser')
    parser = set_parser()
    
    args, unknown = parser.parse_known_args()
    
    # put arguments from (inference) config_file            
    print(f'from_only_config: {args.from_only_config}')
    if args.from_only_config:
        # load (inference) config file
        # TODO 파일 안 열릴 때 종료하는 예외처리
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        with open(config_dict['saved_inference_config_path'], 'r') as f:
            inference_config_dict = json.load(f)

        args = argparse.Namespace()
        d = vars(args)
        for key, value in inference_config_dict.items():
            d[key] = value    
    # add variable to argments for record
    d = vars(args)

    print('=' * 15 + 'arguments for inference' + '=' *15)  
    print(args)
    
    print(f'Inference of test data and save submission file to be ready to submit')

    inference(args)