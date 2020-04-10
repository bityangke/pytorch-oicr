import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.autograd import Variable


def get_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--k_iter', type=int, default=1)
    parser.add_argument('--worker', type=int, default=1)
    
    parser.add_argument('--epoch', type=int, default=20)
    
    parser.add_argument('--datamode', type=str, default='train')
    
    # Directory
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    #parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--image_label_path', default='voc_2007_trainval_image_label.json')
    #parser.add_argument('--ssw_path', default='ssw.txt')
    parser.add_argument('--ssw_path', default='voc_2007_trainval.mat')
    
    parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='Load pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint info')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard info')
    
    # Backbone Network
    parser.add_argument('--backbone_network', type=str, default='vgg16')
    
    args = parser.parse_args()
    
    return args


def train(args, model):
    
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    if args.datamode == 'train':
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, \
                                                    shuffle=False, num_workers=args.worker)
        
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), \
                                                   desc='Train epoch= %d' % epoch, ncols=80, leave=False):
            data  = Variable(data).cuda()
    

def main():
    args = get_args()
    print(args)
    
    


if __name__ == '__main__':
    main()