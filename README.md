# CAM-in-VGG-by-pytorch
show the CAM in some of trained VGG models by pytorch using CIFAR10  
VGG models: VGG9_conv, VGG9_avgpool,  ......  
VGG9_conv means the last classifier layer is conv2d instead of fc  
VGG9_avgpool means before the fc layer, use avgpool first  

When I use two GPUs to train the model, althought the acc is high enough, the results of CAM isn't good as well.  

## Original Images
test2.jpg|test3.jpg|test4.jpg
:---:|:---:|:---:
<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/test%20imgs/test2.jpg" width="150" alt="test2.jpg">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/test%20imgs/test3.jpg" width="150" alt="test3.jpg">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/test%20imgs/test4.jpg" width="150" alt="test4.jpg">|

## CAM Results
### test2.jpg
VGG11_avgpool|VGG11_conv|VGG9_avgpool|VGG9_conv
:---:|:---:|:---:|:---:
<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM2_VGG11_avgpool.jpg" width="150" alt="test2.jpg VGG11_avgpool ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM2_VGG11_conv.jpg" width="150" alt="test2.jpg VGG11_conv ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM2_VGG9_avgpool.jpg" width="150" alt="test2.jpg VGG9_avgpool "/>|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM2_VGG9_conv.jpg" width="150" alt="test2.jpg VGG9_conv "/>
### test3.jpg
VGG11_avgpool|VGG11_conv|VGG9_avgpool|VGG9_conv
:---:|:---:|:---:|:---:
<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM3_VGG11_avgpool.jpg" width="150" alt="test3.jpg VGG11_avgpool ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM3_VGG11_conv.jpg" width="150" alt="test3.jpg VGG11_conv ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM3_VGG9_avgpool.jpg" width="150" alt="test3.jpg VGG9_avgpool "/>|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM3_VGG9_conv.jpg" width="150" alt="test3.jpg VGG9_conv "/>
### test4.jpg
VGG11_avgpool|VGG11_conv|VGG9_avgpool|VGG9_conv
:---:|:---:|:---:|:---:
<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM4_VGG11_avgpool.jpg" width="150" alt="test4.jpg VGG11_avgpool ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM4_VGG11_conv.jpg" width="150" alt="test4.jpg VGG11_conv ">|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM4_VGG9_avgpool.jpg" width="150" alt="test4.jpg VGG9_avgpool "/>|<img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/cam%20results/CAM4_VGG9_conv.jpg" width="150" alt="test4.jpg VGG9_conv "/>

The output classes of test4.jpg are "bird", but it can easy to see that the model didn't learn the features well. I think one of the reason is that the VGGnet is too deep for the CIFAR10 dataset. After convolutional calculation, 3x32x32 images transfered to 512x1x1 features vector.
