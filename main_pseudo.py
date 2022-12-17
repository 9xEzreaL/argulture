import torch
import torch.nn as nn
import os
import argparse
import datetime
import json
from os.path import join
import torch.utils.data as data
from util.Metric import Metric
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
from vision_model import hardnet, densenet, efficientnet, densenet_201, arc_efficientnet, Resnext, Resnest
import torch.optim as optim
from util.sam import SAM
from util.smooth_crossentropy import smooth_crossentropy
from util.by_pass import enable_running_stats, disable_running_stats


"""
Train a classfier model and save it so that you can use this classfier to test your generated data.
"""

attrs_default = [
     'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
     'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
     'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
     'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'
]


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')

    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='# of epochs')
    parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', type=int, default=20) # training batch size
    parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=12,
                        help='# of sample images')  # valid batch size
    parser.add_argument('--net', dest='net', default='vgg')
                        # choices=['vgg', 'inception', 'densenet', 'resnet', 'squ_net', 'vgg11', 'vgg11_mc'])
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=2000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--resize', dest='resize', default=0, type=int, help='image resize')
    parser.add_argument('--ckpt', dest='ckpt', default=None)


    return parser.parse_args(args)



class Classifier():
    def __init__(self, args, net):
        self.args = args
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.model = self.network_map(net)(use_meta=True)
        self.model.train()
        if self.gpu: self.model.cuda()

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        base_optim = optim.SGD
        self.optim_model = SAM(self.model.parameters(), base_optim, lr=args.lr, momentum=0.9)
        if args.fine_tune:
            self.load(args.fine_tune, args.ckpt)


    def set_lr(self, lr):
        for g in self.optim_model.param_groups:
            g['lr'] = lr

    def train_model(self, img_a, label, geo_info, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        for p in self.model.parameters():
            p.requires_grad = True
        enable_running_stats(self.model)
        pred = self.model(img_a, geo_info)
        _, label = label.max(1)
        label = label.type(torch.int64)

        dc_loss = smooth_crossentropy(pred, label, smoothing=0.1)
        dc_loss.mean().backward()
        self.optim_model.first_step(zero_grad=True) # this is optimizer sam

        disable_running_stats(self.model)
        smooth_crossentropy(self.model(img_a, geo_info), label, smoothing=0.1).mean().backward()
        self.optim_model.second_step(zero_grad=True)

        _, predicted = pred.max(1)
        # _, targets = label.max(1)
        metric.update(predicted, label)
        acc = metric.accuracy()
        f1 = metric.f1()

        errD = {
            'd_loss': dc_loss.mean().item()
        }
        return errD, acc, f1

    def eval_model(self, img_a, label, geo_info, metric): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        with torch.no_grad():
            pred = self.model(img_a, geo_info)
        label = label.type(torch.float)

        _, predicted = pred.max(1)
        _, targets = label.max(1)
        metric.update(predicted, targets)
        acc = metric.accuracy()
        f1 = metric.f1()
        each_f1 = metric.f1(each_cls=True)

        return acc, f1, each_f1

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
            'optim_model': self.optim_model.state_dict(),
        }
        torch.save(states, path)

    def load(self, fine_tune=False, ckpt=None):
        if fine_tune:
            states = torch.load(ckpt)
            self.model.load_state_dict(states['model'])
            for module in self.model.modules():
                # print(module)
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
        # states = torch.load(path, map_location=lambda storage, loc: storage)
        # if 'model' in states:
        #     self.model.load_state_dict(states['model'])
        # if 'optim_model' in states:
        #     self.optim_model.load_state_dict(states['optim_model'])

    def network_map(self, net):
        network_mapping = {
            'meta_densenet': densenet,
            'meta_densenet_201': densenet_201,
            'meta_efficientnet': efficientnet,
            'meta_hardnet': hardnet,
            'arc_meta_efficientnet': arc_efficientnet,
            'meta_resnext': Resnext,
            'meta_resnest': Resnest
        }
        return network_mapping[net]

if __name__=='__main__':
    dst_root = '/media/ExtHDD01/argulture_log/' # where your model saved
    data_root = '/media/ExtHDD01/Dataset/argulture_all/' # where your data root in
    csv_path = '/media/ExtHDD01/Dataset/argulture/ensemble.csv' # where your csv

    args = parse()

    if args.ckpt is not None:
        args.fine_tune=True
        args.ckpt = os.path.join(dst_root, args.ckpt)
    else:
        args.fine_tune=False
    args.lr_base = args.lr
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)

    os.makedirs(join(dst_root, args.experiment_name), exist_ok=True)
    os.makedirs(join(dst_root, args.experiment_name, 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(dst_root, args.experiment_name, 'summary'))

    with open(join(dst_root, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    from dataloader import MetaDataset
    train_dataset = MetaDataset(data_root, csv_path, 'train', 0)
    valid_dataset = MetaDataset(data_root, csv_path, 'valid', 0)
    test_dataset = MetaDataset(data_root, csv_path, 'test', 0)

    num_gpu = torch.cuda.device_count()

    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size_per_gpu * num_gpu,
                                       num_workers=10, drop_last=True, sampler=ImbalancedDatasetSampler(train_dataset))
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size_per_gpu * num_gpu, shuffle=False, num_workers=10, drop_last=False)

    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

    classifier = Classifier(args, net=args.net)
    progressbar = Progressbar()

    it = 0
    it_per_epoch = len(train_dataset) // (args.batch_size_per_gpu * num_gpu)
    for epoch in range(args.epochs):
        lr = args.lr_base * (0.9 ** epoch)
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        metric_tr = Metric(num_classes=33)
        metric_ev = Metric(num_classes=33)
        for img_a, att_a, geo_info, _ in progressbar(train_dataloader):
            img_a = img_a.cuda() if args.gpu else img_a
            att_a = att_a.cuda() if args.gpu else att_a
            geo_a = geo_info.cuda() if args.gpu else geo_info
            att_a = att_a.type(torch.float)
            img_a = img_a.type(torch.float)
            geo_a = geo_a.type(torch.float)

            errD, acc, f1 = classifier.train_model(img_a, att_a, geo_a, metric_tr)
            add_scalar_dict(writer, errD, it+1, 'D')
            it += 1
            progressbar.say(epoch=epoch, d_loss=errD['d_loss'], acc=acc.item(), f1=f1.item())
        classifier.save(os.path.join(
            dst_root, args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        ))


# CUDA_VISIBLE_DEVICES=3 python main_first.py --net meta_densenet --experiment_name meta_densenet_sam_optim_512_1028_1030 --lr 0.0005 --ckpt meta_densenet_sam_optim_512_1028/checkpoint/weights.29.pth --gpu --batch_size 15