from utils.CAM import *
# from CAM2 import *
from torchvision import models, transforms
import torch
from cifar10.classifiers.vgg import *


model = VGG11_avgpool()

# model = VGG9_avgpool()
pretrained = True
img_path = './test4.jpg'
# save_path = './CAM4test.jpg'
label_path = './cifar10_labels.json'
checkpoint_path = './checkpoint/cifar10_gpu_50_VGG11_avgpool.pth'

cam = CAM(model, True , checkpoint_path,'cpu', img_path, label_path, 'CAM4_VGG11_avgpool')




# normalize = transforms.Normalize(
# mean=[0.485, 0.456, 0.406],
# std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
# transforms.Resize((32,32)),
# transforms.ToTensor(),
# normalize
# ])

# transforms = preprocess

# cam = draw_CAM(model, img_path,  save_path, transform=transforms, visual_heatmap=True)
