# coding: UTF-8
import time
import torch
import numpy as np
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.data.network1 import Net
from torch.utils.data import DataLoader, DistributedSampler

from models.FastText import Model
from train_eval import train, init_network
from importlib import import_module
import argparse

import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser(description='Chinese Text Classification')
#parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')



parser.add_argument('--device_ids',type=str,default='0',help="Training Devices")
parser.add_argument('--local_rank',type=int,default=-1,help="DDP parameter,do not modify")



args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    # model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer

    model_name = 'FastText'
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)

    # 单机单卡GPU
    model = x.Model(config).to(config.device)  #  原来的形式


    #单机单进程多卡GPU
    # gpus = [0, 1, 2, 3,4,5,6,7]
    # model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])


    '''
    #单机多进程多卡GPU    python -m torch.distributed.launch --nproc_per_node 4 --master_port 8005 run.py --device_ids=4,5,6,7
    device_ids = list(map(int,args.device_ids.split(',')))
    dist.init_process_group(backend='nccl',init_method='env://')
    device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    torch.cuda.set_device(device)
    model = x.Model(config).to(device)
    model = DistributedDataParallel(model,device_ids=[device_ids[args.local_rank]],output_device=device_ids[args.local_rank])

    '''





    # 多块GPU
    #model = x.Model(config)#.to(config.device)
    #model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()


    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
