import argparse
import json
import requests
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import os
from importlib import import_module, reload







def submit(user_key='', file_path = '', desc=''):
    if not user_key:
        raise Exception("No UserKey" )
    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/35/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': user_key
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000035/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})
    
    


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

    
    if args.submission_user_key != '':
        user_key = args.submission_user_key
        output_file_path = os.path.join(args.submission_dir, f'{args.saved_model_name}.csv')
        desc = f'model:{args.model}, backbone:{args.backbone}, neck:{args.neck} optimizer:{args.optimizer}, scheduler:{args.scheduler} lr:{args.lr}\n' \
        f'epoch:{args.best_epoch}/{args.epochs}, best loss:{args.best_loss}, batch size:{args.batch_size}\n' \
        f'train augmentation:{args.train_augmentation}, val augmentation:{args.val_augmentation}, test augmentation:{args.test_augmentation}'
        submit(user_key, output_file_path, desc)
    
    