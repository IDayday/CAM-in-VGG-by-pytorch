from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch
import json
import cv2
import pdb


def model(model_id, pretrained):
    if model_id == 1:
        net = models.squeezenet1_1(pretrained=pretrained)
        finalconv_name = 'features' # this is the last conv layer of the network
    elif model_id == 2:
        net = models.resnet18(pretrained=pretrained)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=pretrained)
        finalconv_name = 'features'
    net.eval()
    print(net)
    return net, finalconv_name
"""
def register_forward_hook(self, hook):

       handle = hooks.RemovableHandle(self._forward_hooks)
       self._forward_hooks[handle.id] = hook
       return handle
这个方法的作用是在此module上注册一个hook，函数中第一句就没必要在意了，主要看第二句，是把注册的hook保存在_forward_hooks字典里。
hook 只能注册到 Module 上，即，仅仅是简单的 op 包装的 Module，而不是我们继承 Module时写的那个类，我们继承 Module写的类叫做 Container

当我们执行model(x)的时候，底层干了以下几件事：

1.调用 forward 方法计算结果
2.判断有没有注册 forward_hook，有的话，就将 forward 的输入及结果作为hook的实参。然后让hook自己干一些不可告人的事情。
"""

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def CAM(model, pretrained, img_path, label_path, out_name):
    if model == int:
        net, finalconv_name = model(model, pretrained)
    else:
        net = model
        finalconv_name = 'features'
        net.eval()
        print(net)
        if pretrained ==True:
            net.load_state_dict(torch.load('./checkpoint/cifar10_cpu_150.pth'))
            net.eval()
    for name, module in net.named_modules():
        print('modules:', name)
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
    ])

    # img_path = './test2.jpg'

    img_pil = Image.open(img_path)
    img_tensor = preprocess(img_pil).unsqueeze(0)
    # img_variable = Variable(img_tensor.unsqueeze(0))

    logit = net(img_tensor)

    json_path = label_path
    # json_path = './labels.json'
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
    classes = {int(key): value for (key, value)
            in load_json.items()}

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output' + out_name + '.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(img_path[2:])
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(out_name + '.jpg', result)
