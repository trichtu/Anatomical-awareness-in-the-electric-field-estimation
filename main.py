import argparse
import os
from solver import Solver
from data_loader import get_data_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['two_step','U_Net_Seg','U_Net3D','U_Net3D_V2','U_Net3D_Att','U_Net3D_Att_V2','U_Net3D_Att_Seg','U_Net3D_Att_Seg_V2']:
        print('ERROR!! model_type should be selected in two_step/U_Net_Seg/U_Net3D_Att/U_Net3D_Att_V2/U_Net3D_Att_Seg/U_Net3D_Att_Seg_V2')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = 0.0005
    augmentation_prob = 0
    epoch = 70
    decay_ratio = 0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch
    dis = config.dis
    batch = config.batch_size

    train_loader = get_data_loader(batch, 4, 'train' )
    val_loader = get_data_loader(batch, 4, 'val' )
    test_loader = get_data_loader(batch, 4, 'train' )

    solver = Solver(config, train_loader, val_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'evaluation':
        solver.evaluation()
    elif config.mode == 'seg':
        solver.segmentation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    # parser.add_argument('--image_size', type=int, default=224)
    # parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    # parser.add_argument('--num_epochs', type=int, default=30)
    # parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.1)
    parser.add_argument('--test_only', type=bool, default=False)
    parser.add_argument('--dis', type=bool, default=False)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--historyhour', type=int, default=2)
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net3D_Att', help='U_Net3D/U_Net3D_Att/U_Net3D_Att_Seg/U_Net3D_Tissue_Att')
    parser.add_argument('--model_path', type=str, default='./models')
    # parser.add_argument('--train_path', type=str, default='./dataset/train/')
    # parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    # parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
