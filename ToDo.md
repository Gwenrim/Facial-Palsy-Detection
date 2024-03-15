# 指令

\#TODO:

\1. 可以修改的地方：loss，attention；

\2. 需测试landmarks，是全部用上，还是用部分差值；

\3. 看看tensorboard如何远程用；

\4. 看看如何整个数据上传服务器；

\5. distillation，Teacher--Student；

\6. 视频流和特征点流考虑是否中途合并；【待测试】

\7. ROC指标；

\8. 正常人数据下载、上传、预处理；

***\*8. 数据处理：根据特征点，划分patch；【可加可视化图】；再根据不同分支的权重做数据增强\****

Done:

\1. 修改loss；(bs*action_num)处理；preds.T, label.T，4个criterion；

\2. 修改metrics（每个分支的每个类别？），得写个util_func；

\3. 不用marlin，换facenet-pytorch的InceptionResNet；

23/08/28:

\1. video_frames线的feature extractor换成facenet_torch的InceptionResnetV1；（可能需要换mtcnn to crop）

\2. + GRU；【参数不确定】

2023/08/30：

\1. cross-attention实现；

2023/09/11：

\1. check帧数、parsed imgs；

\2. gw01: normal; gw02: parsed imgs;

2023/09/12：

\1. 区域划分，可视化；

2023/09/15

\0. 发现之前动作和分支对应错了，已更正；（注意之后的处理）

\1. 数据预处理，各分支权重；【需确认:2个分支没有第6级】

2023/09/17：

\1. DATASET：扩充策略；

2023/09/18：

1. DATASET: 图像和特征点对应标签处理，size如何统一 ，暂时都224了;
2. 网络视频流特征提取器小修；
3. 上传git；
4. lr=1e-3不行；

特征可视化
ROC

2023/09/29：

1. 混杂后：分支单独实验：【指标很低！！！】
2. 混杂前：分支单独实验；
// 1: branch-1; 2: branch-2; 3: branch-3;
// 4: branch-1+3; 5: branch-2+3;  6: branch-1+2;
// 7: branch-1+2+3;

2023/10/08：

1. 单图预测check；
2. 指标check；
3. 同类任务参考；

\########################################

*** 【loss设计】，loss_weights

*** 预测图像和Label check，对应结果保存可视化，一一对应，看是不是指标预测有问题；

多属性分类任务参考，人体属性识别；标签之间差异性？、设计？标签格式；

相似任务baseline，

！！单分支就有问题，先图像，再考虑序列

搜索记录！！

Excel表记录；

\########################################

数据在读取时就做好shuffle，而不是分训练测试后再做shuffle；

最好用一个DATASET，参数决定是否 Mix
