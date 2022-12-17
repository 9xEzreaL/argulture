import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import cv2
import os
import csv
from PIL import Image
import argparse
from util.Metric import Metric
from dataloader import TestingMetaDataset
from vision_model import efficientnet, densenet_201, densenet, Resnext


attrs_default = [
     'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
     'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
     'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
     'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--use_meta', dest='use_meta', action='store_true')
    parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', type=int, default=20) # training batch size
    parser.add_argument('--net', dest='net', default='vgg')
    parser.add_argument('--net2', dest='net2', default=None)
    parser.add_argument('--net3', dest='net3', default=None)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--log_name', dest='log_name')
    parser.add_argument('--log_name2', dest='log_name2')
    parser.add_argument('--log_name3', dest='log_name3')
    parser.add_argument('--epoch', dest='epoch')
    parser.add_argument('--epoch2', dest='epoch2')
    parser.add_argument('--epoch3', dest='epoch3')

    return parser.parse_args(args)

def network_map(net):
    network_mapping = {
        'meta_densenet': densenet(use_meta=True),
        'meta_densenet_201': densenet_201(use_meta=True),
        'meta_efficientnet': efficientnet(use_meta=True),
        'efficientnet': efficientnet(),
    }
    return network_mapping[net]

def rot90(img, clockwise = True):
    img = img.detach().cpu()
    img = np.array(img)
    if img.shape[0]==3:
        img = np.transpose(np.array(img),(1,2,0))
    img_rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # img_rot90 = torch.rot90(img)
    img_rot90 = np.transpose(img_rot90, (2, 0, 1))
    return torch.Tensor(img_rot90).cuda()

def flip(img, dic='h'):
    img = img.detach().cpu()
    img = np.array(img)
    if img.shape[0]==3:
        img = np.transpose(np.array(img),(1,2,0))

    if dic == 'h':
        fliped_img = cv2.flip(img, 1)
    elif dic == 'v':
        fliped_img = cv2.flip(img, 0)
    fliped_img = np.transpose(fliped_img, (2,0,1))
    return torch.Tensor(fliped_img).cuda()

def schedule(model, img, meta, model2, model3):
    final_out = torch.empty(1, 33).cuda()
    for i in range(img.shape[0]):
        o_img = img[i,::]
        o_meta = meta[i, ::]
        o_meta = torch.unsqueeze(o_meta, 0)
        h_img = flip(o_img, 'h')
        v_img = flip(o_img, 'v')
        hv_img = flip(v_img, 'h')
        o_img = torch.unsqueeze(o_img, 0)
        h_img = torch.unsqueeze(h_img, 0)
        v_img = torch.unsqueeze(v_img, 0)
        hv_img = torch.unsqueeze(hv_img, 0)

        r_img = rot90(img[i,::])
        rh_img = flip(r_img, 'h')
        rv_img = flip(r_img, 'v')
        rvh_img = flip(rv_img, 'h')
        r_img = torch.unsqueeze(r_img, 0)
        rh_img = torch.unsqueeze(rh_img, 0)
        rv_img = torch.unsqueeze(rv_img, 0)
        rvh_img = torch.unsqueeze(rvh_img, 0)

        batch1_img = torch.cat([o_img, h_img, v_img, hv_img, r_img, rh_img, rv_img, rvh_img], 0).cuda()
        o_meta = torch.cat([o_meta]*8, 0).cuda()
        out = model(batch1_img, o_meta)
        out[:, 19] = out[:,19] #+0.2
        if model2 is not None:
            try:
                out2 = model2(batch1_img)
            except:
                out2 = model2(batch1_img, o_meta)
            out2[:, 19] = out2[:, 19]+0
            out += out2
        if model3 is not None:
            try:
                out3 = model3(batch1_img)
            except:
                out3 = model3(batch1_img, o_meta)
            out3[:, 19] = out3[:, 19]+0
            out += out3
        out = torch.unsqueeze(torch.sum(out, 0), 0)
        final_out = torch.cat([final_out, out], 0)
    return final_out[1:,:]



if __name__=='__main__':
    root = '/media/ExtHDD01/argulture_log'
    data_root = '/media/ExtHDD01/Dataset/argulture_test/'
    csv_path = '/media/ExtHDD01/Dataset/argulture_test/test.csv'
    ckpt = 'checkpoint'
    args = parse()

    test_dataset = TestingMetaDataset(data_root, csv_path, 'test')
    num_gpu = torch.cuda.device_count()
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size_per_gpu * num_gpu, shuffle=False, num_workers=10, drop_last=False)
    print('Training images:', len(test_dataset))

    model = network_map(args.net).cuda()
    states = torch.load(os.path.join(root, args.log_name, ckpt, f'weights.{args.epoch}.pth'))
    model.load_state_dict(states['model'])
    model.eval()

    if args.net2 is not None:
        model2 = network_map(args.net2).cuda()
        states = torch.load(os.path.join(root, args.log_name2, ckpt, f'weights.{args.epoch2}.pth'))
        model2.load_state_dict(states['model'])
        model2.eval()
    else:
        model2 = None
    if args.net3 is not None:
        model3 = network_map(args.net3).cuda()
        states = torch.load(os.path.join(root, args.log_name3, ckpt, f'weights.{args.epoch3}.pth'))
        model3.load_state_dict(states['model'])
        model3.eval()
    else:
        model3 = None

    id_t = ()
    pred_t = []
    for img, meta, id in test_dataloader:
        id_t += id
        img = img.cuda() if args.gpu else img
        meta = meta.cuda() if args.gpu else meta
        img = img.type(torch.float)
        meta = meta.type(torch.float)
        with torch.no_grad():
            pred = schedule(model, img, meta, model2, model3)
            # pred = model(img)
            _, predicted = pred.max(1)

            pred_t.append(predicted)

    pred = torch.cat(pred_t, 0)

    # document results
    os.makedirs('result', exist_ok=True)
    with open(f'result/{args.log_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i in range(len(pred)):
            tmp = [id_t[i], attrs_default[pred[i].item()]]
            writer.writerow(tmp)

# CUDA_VISIBLE_DEVICES=0 python test.py --log_name efficientnet_512_batch15_CEloss_1008 --epoch 26 --gpu --net efficientnet
# CUDA_VISIBLE_DEVICES=2 python test.py --net efficientnet --log_name efficientnet_640_batch7_CEloss_1006 --epoch 25 --batch_size 60 --gpu