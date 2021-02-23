import torch
import os;
import numpy as np
import cv2
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
from torch.nn import functional as F
import requests


if __name__=="__main__":
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    image_path="./test.jpg";
    img_pil = Image.open(image_path);
    img_tensor = preprocess(img_pil);# 返回一个三维的tensor
    img_variable = Variable(img_tensor.unsqueeze(0));# 返回一个4维的tensor, 第一个维度是1, 相当于batchsize

    net = models.resnet50(pretrained=True);
    finalconv_name = 'layer4';
    net.eval()# 开起test 模式
    logit = net(img_variable);
    h_x = F.softmax(input=logit, dim=1).data.squeeze();
    # 图片的四维tensor 作为input, output是 1000个分类的 probability score, 应用softmax作为loss function
    probs, idx = h_x.sort(dim=0, descending=True);#排序, 最大值在第一个, 它的位置就是label
    probs = probs.numpy();
    idx = idx.numpy();

    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    classes = {int(key): value for (key, value)
               in requests.get(LABELS_URL).json().items()}

    print("The label is ", idx[0], ", The corresponding things is", classes[idx[0]]);


