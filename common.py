import argparse

'''
parser
'''


def set_parser():
    
    def str2bool(str):
        if str == 'True':
            return True
        return False
    
    parser = argparse.ArgumentParser()
    # from_only_config ignores all argments
    parser.add_argument('--from_only_config', type=str2bool, default=False, help='it loads argments only from config.json (default: False)')
    parser.add_argument('--config_path', type=str, default='/opt/ml/code/config.json', help='config_path (default: /opt/ml/code/config.json)')

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset (default: CustomDataset)')
    parser.add_argument('--dataset_dir', type=str, default='/opt/ml/input/data', help='dataset_dir (default: /opt/ml/input/data)')
    parser.add_argument('--train_augmentation', type=str, default='BaseAugmentation', help='train_augmentation (default: BaseAugmentation)')
    parser.add_argument('--val_augmentation', type=str, default='BaseAugmentation', help='val_augmentation (default: BaseAugmentation)')
    parser.add_argument('--test_augmentation', type=str, default='TestAugmentation', help='test_augmentation (default: TestAugmentation)')
    parser.add_argument('--model', type=str, default='FCN8s', help='model (default: FCN8s)')
    parser.add_argument('--encoder', type=str, default='resnet101', help='encoder (default: resnet101)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='lr (default: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='', help='scheduler (default: '')')
    parser.add_argument('--scheduler_parameter', type=dict, default={}, help='scheduler_parameter (default: {})')        
        
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size (default: 16)')
    parser.add_argument('--random_seed', type=int, default=21, help='random_seed (default: 21)')
    parser.add_argument('--epochs', type=int, default=20, help='epochs (default: 20)')
    parser.add_argument('--val_every', type=int, default=1, help='val_every (default: 1)')
    
    parser.add_argument('--saved_model_name', type=str, default='saved_model_best', help='saved_model_name (default: saved_model_best)')
    parser.add_argument('--saved_inference_config_path', type=str, default='/opt/ml/code/inference_config.json', help='saved_inference_config_path (default: /opt/ml/code/inference_config.json)')
    parser.add_argument('--saved_dir', type=str, default='/opt/ml/code/saved', help='saved_dir (default: /opt/ml/code/saved)')
    parser.add_argument('--submission_dir', type=str, default='/opt/ml/code/submission', help='submission_dir (default: /opt/ml/code/submission)')
    parser.add_argument('--submission_user_key', type=str, default='', help='submission_user_key (default: '')')
    
    parser.add_argument('--is_wandb', type=int, default=1, help='is_wandb (default: 1)')
    parser.add_argument('--wandb_project_name', type=str, default='pstage3_image_segmentation', help='wandb_project_name (default: pstage3_image_segmentation)')
    parser.add_argument('--wandb_group', type=str, default='experiments_group_name', help='wandb_group (default: experiments_group_name)')
    parser.add_argument('--wandb_experiment_name', type=str, default='', help='wandb_experiment_name (default: '')')

    return parser
