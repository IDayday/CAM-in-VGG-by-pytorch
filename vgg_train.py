import torch.optim as optim
import torch.nn as nn
import time
import os
import datetime
from cifar10.tnt_solver import *
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from cifar10.classifiers.vgg import *
import torchnet.meter as meter
import torch
import pandas as pd
import json
from torch.utils.tensorboard import SummaryWriter
# 批训练管理
class RunManager():
    #导入数据，设置网络，设置参数，记录训练时间，记录循环周期，记录追踪目标，记录tensorboard日志
    def __init__(self):
        self.loader = None
        self.network = None
        self.params = None

        self.run_start_time = time.time()
        self.epoch_start_time = None
        self.run_start_time = None

        self.run_count = 0
        self.epoch_count = 0

        self.epoch_num_correct = 0
        self.epoch_loss = 0
        self.run_data = []

        self.tb = None

    def begin_run(self, run, network, loader, val_loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.loader = loader
        self.val_loader = val_loader
        self.network = network
        # 迭代器next()传入数据，按照loader的设置分批传入
        # images, labels = next(iter(self.loader))
        # images = images.to(device)
        # labels = labels.to(device)
        # grid = torchvision.utils.make_grid(images)

        # f''格式化字符串，{}中表示被替换的内容
        # SummaryWriter是给运行日志命名
        self.tb = SummaryWriter(comment=f'-{run}')
        # self.tb.add_image('images',grid)
        # 添加图时，既要有网络，也要有输入
        # if len(run.gpus) > 1:
            # self.tb.add_graph(
                # .module是将模型从Dataparallel中取出来再写入tensorboard，否则并行时会报错。
                # self.network.module,
                # getattr获得device属性值，没有就输出默认值cpu，都没有则会报错
                # images.to(getattr(run, 'device','cpu'))
                # images.to('cuda')
            # )
        # else:
            # self.tb.add_graph(self.network,images.to('cuda'))
    def end_run(self):
        self.tb.close()
        # 一个epoch还没有结束，所以epoch_count不计数
        self.epoch_count = 0
    
    def begin_epoch(self):
        # 初始化本轮epoch中的一些记录点
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_num_correct = 0
        self.epoch_loss = 0

    # 测试集准确率，观察防止过拟合。一般应该使用验证集
    def val(self, network, val_loader):
        network.eval()
        num_class = 10
        confusion_matrix = meter.ConfusionMeter(num_class)
        for ii, data in enumerate(val_loader):
            images, labels = data
            with torch.no_grad():
                val_images = images
                val_labels = labels
            if torch.cuda.is_available():
                val_images = val_images.cuda()
                val_labels = val_labels.cuda()
            score = network(val_images)
            confusion_matrix.add(score.data.squeeze(), labels.long())

        # 把模型恢复为训练模式
        network.train()

        cm_value = confusion_matrix.value()
        error_sum = 0
        for i in range(num_class):
            error_sum += cm_value[i][i]
        test_accuracy = error_sum / (cm_value.sum())
        return confusion_matrix, test_accuracy

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss/len(self.loader.dataset)
        accuracy = self.epoch_num_correct/len(self.loader.dataset)
        confusion_matrix, test_acc = self.val(self.network, self.val_loader) 

        # add_scalar给tensorboard添加标量数据，对应'名称'，'Y轴','X轴'
        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        self.tb.add_scalar('Test_Accuracy', test_acc, self.epoch_count) 
        print('epoch:',self.epoch_count,'Loss', loss, 'Accuracy', accuracy,'Test_Accuracy',test_acc)
        
        # tensorboard记录网络权重及权重梯度
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        # 训练日志设置
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)


    def track_loss(self, loss):
        self.epoch_loss += loss.item()*self.loader.batch_size
    
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    # 统计一个批次内的正确个数和损失
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        # 把run_data列表里的有序字典按列存储，也就是名称在顶部，依次向下排。
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns').to_csv(f'{fileName}.csv')
        with open(f'{fileName},json','w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

# 调参库
class RunBuilder():
    # @staticmethod目的是获得静态方法并且不需要实例化
    @staticmethod
    # 返回的是一个list，里面包含着重组后的具名元组
    def get_runs(params):
        # Run是一个具名元组的方法，会将传入的参数依次对应到设置的名称下。
        Run = namedtuple('run',params.keys())
        runs = []
        # 参数重组
        # *表示自动对应多个参数
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


params = OrderedDict(
    lr = [.001],
    batch_size = [256],
    shuffle = [True],
    gpus = ['0,1'],
    num_worker = [4],
    train_split = [0.8],
    stratify = [False],
    # model = ['VGG9_conv', 'VGG11_conv', 'VGG17_conv']
    model = ['VGG9_avgpool', 'VGG11_avgpool', 'VGG17_avgpool']
)

# 训练循环主体
rm = RunManager()

model_name = [VGG9_avgpool(), VGG11_avgpool(), VGG17_avgpool(), VGG9_conv(), VGG11_conv(), VGG17_conv()]
C = CIFAR10Data()
for run in RunBuilder.get_runs(params):
    # run依次获得list中的各个具名元组，所以可以将名称作为属性直接调出例如run.batch_size
    print(run)
    os.environ["CUDA_VISIBLE_DEVICES"] = run.gpus
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load = C.data_split(run.train_split, run.stratify)
    loader = C.get_train_loader(batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_worker)
    val_loader = C.get_test_loader(batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_worker)
    
    if run.model == 'VGG9_avgpool':
        network = model_name[0]
    elif run.model == 'VGG11_avgpool':
        network = model_name[1]
    elif run.model == 'VGG17_avgpool':
        network = model_name[2]
    elif run.model == 'VGG9_conv':
        network = model_name[3]
    elif run.model == 'VGG11_conv':
        network = model_name[4]
    elif run.model == 'VGG17_conv':
        network = model_name[5]

    if len(run.gpus)>1:
        network = nn.DataParallel(network)
        print('DataParallel already!')
    network = network.to(device)
    network.train()
    num_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    rm.begin_run(run, network, loader, val_loader)
    for epoch in range(50):
        train_loss = 0
        correct = 0
        total = 0
        rm.begin_epoch()
        epoch_start = time.time()
        for batch_idx, (images, labels) in enumerate(loader):
            start = time.time()

            images = images.to(device)
            labels = labels.to(device)
            if run.model in ('VGG9_conv', 'VGG11_conv', 'VGG17_conv'):
                pred = network(images)
                # print(preds)
                preds = torch.squeeze(pred)
                # print(preds)
            else:
                preds = network(images)
                # print(preds)
            # 实例化损失函数才能使用
            loss = loss_fn(preds,labels)
            # loss = loss.to(device)
            # 总得来说，这三个函数的作用是先将梯度归零optimizer.zero_grad()
            # 然后反向传播计算得到每个参数的梯度值loss.backward()
            # 最后通过梯度下降执行一步参数更新optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = preds.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            acc = 100 * correct / total

            rm.track_loss(loss)
            rm.track_num_correct(preds, labels)

            batch_time = time.time() - start
            if batch_idx % 20 == 0:
                print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(loader), train_loss/(batch_idx+1), acc, batch_time))
        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("Training time {}".format(elapse_time))
        rm.end_epoch()
    rm.end_run()
    torch.save(network.state_dict(), "./checkpoint/cifar10_gpu_50_"+ run.model+ ".pth")
rm.save('results_cifar10')
