# CAM-in-VGG-by-pytorch
show the CAM in some of trained VGG models by pytorch  
VGG models: VGG9_conv, VGG9_avgpool,  ......  
VGG9_conv means the last classifier layer is conv2d instead of fc  
VGG9_avgpool means before the fc layer, use avgpool first  

When I use two GPUs to train the model, althought the acc is high enough, the results of CAM isn't good as well.  

VGG11_avgpool...........VGG11_conv..........VGG9_avgpool..........VGG9_conv
<div align=center><img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/CAM4_VGG11_avgpool.jpg" width="150" alt="test4.jpg VGG11_avgpool "><img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/CAM4_VGG11_conv.jpg" width="150" alt="test4.jpg VGG11_conv "><img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/CAM4_VGG9_avgpool.jpg" width="150" alt="test4.jpg VGG9_avgpool "/><img src="https://github.com/IDayday/CAM-in-VGG-by-pytorch/blob/main/CAM4_VGG9_conv.jpg" width="150" alt="test4.jpg VGG9_conv "/></div>

The four output classes are "bird", but it can easy to see that the model didn't learn the features well.
