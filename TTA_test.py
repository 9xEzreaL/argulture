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
from main import VGG13, resnet, inception_model, VGG11
from vision_model import efficientnet, densenet_201, densenet, Resnext, Resnest


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
    parser.add_argument('--net4', dest='net4', default=None)
    parser.add_argument('--net5', dest='net5', default=None)
    parser.add_argument('--net6', dest='net6', default=None)
    parser.add_argument('--net7', dest='net7', default=None)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--log_name', dest='log_name')
    parser.add_argument('--log_name2', dest='log_name2')
    parser.add_argument('--log_name3', dest='log_name3')
    parser.add_argument('--log_name4', dest='log_name4')
    parser.add_argument('--log_name5', dest='log_name5')
    parser.add_argument('--log_name6', dest='log_name6')
    parser.add_argument('--log_name7', dest='log_name7')
    parser.add_argument('--epoch', dest='epoch')
    parser.add_argument('--epoch2', dest='epoch2')
    parser.add_argument('--epoch3', dest='epoch3')
    parser.add_argument('--epoch4', dest='epoch4')
    parser.add_argument('--epoch5', dest='epoch5')
    parser.add_argument('--epoch6', dest='epoch6')
    parser.add_argument('--epoch7', dest='epoch7')
    return parser.parse_args(args)

def network_map(net):
    network_mapping = {
        'meta_densenet': densenet,
        'meta_densenet_201': densenet_201,
        'meta_efficientnet': efficientnet,
        'meta_resnest': Resnest
    }
    net = network_mapping[net](use_meta=True)
    return net

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

def schedule(model, img, meta, model2=None, model3=None):
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
        sfm = nn.Softmax(dim=1)
        out = model(batch1_img, o_meta)
        out = sfm(out)
        out[:, 4] = out[:,4] + 0.2
        # out[:, 19] = out[:,19] + 0.1
        if model2 is not None:
            try:
                out2 = model2(batch1_img)
            except:
                out2 = model2(batch1_img, o_meta)
            out2 = sfm(out2)
            out[:, 4] = out[:, 4] + 0.2
            # out2[:, 19] = out2[:, 19]+0.1
            out += out2

        if model3 is not None:
            try:
                out3 = model3(batch1_img)
            except:
                out3 = model3(batch1_img, o_meta)
            out3 = sfm(out3)
            out[:, 4] = out[:, 4] + 0.2
            # out3[:, 19] = out3[:, 19]+0.1
            out += out3
        out = torch.unsqueeze(torch.sum(out, 0), 0)
        final_out = torch.cat([final_out, out], 0)
    return final_out[1:,:]



if __name__=='__main__':
    root = '/media/ExtHDD01/argulture_log'
    data_root = '/media/ExtHDD01/Dataset/p_t/'
    csv_path = '/media/ExtHDD01/Dataset/p_t/test.csv'
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
    if args.net4 is not None:
        model4 = network_map(args.net4).cuda()
        states = torch.load(os.path.join(root, args.log_name4, ckpt, f'weights.{args.epoch4}.pth'))
        model4.load_state_dict(states['model'])
        model4.eval()
    else:
        model4 = None
    if args.net5 is not None:
        model5 = network_map(args.net5).cuda()
        states = torch.load(os.path.join(root, args.log_name5, ckpt, f'weights.{args.epoch5}.pth'))
        model5.load_state_dict(states['model'])
        model5.eval()
    else:
        model5 = None
    if args.net6 is not None:
        model6 = network_map(args.net6).cuda()
        states = torch.load(os.path.join(root, args.log_name6, ckpt, f'weights.{args.epoch6}.pth'))
        model6.load_state_dict(states['model'])
        model6.eval()
    else:
        model6 = None
    if args.net7 is not None:
        model7 = network_map(args.net7).cuda()
        states = torch.load(os.path.join(root, args.log_name7, ckpt, f'weights.{args.epoch7}.pth'))
        model7.load_state_dict(states['model'])
        model7.eval()
    else:
        model7 = None




    sfm = nn.Softmax(dim=1)
    p = torch.zeros((22308, 33))
    p_m = torch.zeros((22308, 33))
    for epoch in range(15):
        print(epoch)
        pred_t = []
        pred_max = []
        id_t = ()
        for img, meta, id in test_dataloader:

            id_t += id
            img = img.cuda() if args.gpu else img
            meta = meta.cuda() if args.gpu else meta
            img = img.type(torch.float)
            meta = meta.type(torch.float)
            with torch.no_grad():
                # pred = schedule(model, img, meta, model2, model3)
                pred = model(img, meta)
                pred = sfm(pred)

                pred2 = model2(img, meta)
                pred2 = sfm(pred2)
                pred2[:, 19] = pred2[:, 19] + 0.1
                pred2[:, 4] = pred2[:, 4] + 0.1
                pred3 = model3(img, meta)
                pred3 = sfm(pred3)
                pred4 = model4(img, meta)
                pred4 = sfm(pred4)
                pred5 = model5(img, meta)
                pred5 = sfm(pred5)
                pred6 = model6(img, meta)
                pred6 = sfm(pred6)
                pred7 = model7(img, meta)
                pred7 = sfm(pred7)*0.5
                # _, predicted = pred.max(1)
                    #
                    # pred_t.append(predicted)
                pred = pred + pred2 + pred3 + pred4 + pred5 + pred6 + pred7
                pred_t.append(pred)
                pred_max.append(pred/6.5)

        pred = torch.cat(pred_t, 0)
        pred_max = torch.cat(pred_max, 0)
        p += pred.detach().cpu()
        p_m += pred_max.detach().cpu()
    _, predicted = p.max(1)
    p_m = p_m / 15 # 30times
    # pred_max = torch.cat(pred_max, 0)
    # document results
    os.makedirs('result', exist_ok=True)
    with open(f'result/{args.log_name}_prob.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage',
     'chinesechives', 'custardapple', 'grape', 'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce',
     'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 'passionfruit', 'pear', 'pennisetum',
     'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo'])
        for i in range(len(pred)):
            tmp = [id_t[i]] + [x.item() for x in p_m[i]]
            writer.writerow(tmp)
    with open(f'result/{args.log_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i in range(len(predicted)):
            tmp = [id_t[i], attrs_default[predicted[i].item()]]
            writer.writerow(tmp)

# CUDA_VISIBLE_DEVICES=0 python test.py --log_name efficientnet_512_batch15_CEloss_1008 --epoch 26 --gpu --net efficientnet
# CUDA_VISIBLE_DEVICES=2 python test.py --net efficientnet --log_name efficientnet_640_batch7_CEloss_1006 --epoch 25 --batch_size 60 --gpu
# CUDA_VISIBLE_DEVICES=1 python test_prob.py --net meta_efficientnet --log_name efficient_geo_768_1016_1122 --batch_size_per_gpu 50 --epoch 23 --gpu --log_name2 meta_densenet_sam_optim_512_1028_1103_1117 --epoch2 28 --net2 meta_densenet


# CUDA_VISIBLE_DEVICES=1 python test_prob.py --net meta_efficientnet --log_name efficient_geo_768_1016_1122 --batch_size_per_gpu 50 --epoch 23 --gpu --log_name2 meta_densenet_all_sam_optim_768_1028_1103 --epoch2 28 --net2 meta_densenet --log_name3 efficientnet_geo_all_720_1021 --epoch3 22 --net3 meta_efficientnet