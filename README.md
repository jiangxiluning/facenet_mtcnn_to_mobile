# FaceNet 和 MTCNN 转 TFLITE

```
pipenv install  # 布道 pipenv , 通过使用 pipenv 安装所有依赖包，使用其他版本的包，有可能出现各种转换问题。
```

## 转换 FaceNet

```
cd facenet
pipenv shell # 孵化出运行项目的 shell 环境，以下命令需要在该环境中运行
```
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

将预训练模型 `20180402-114759` 下载并解压至 `model_pc`，如下图所示：

![WX20190108-113436@2x.png](https://i.loli.net/2019/01/08/5c341f9c5f908.png)


 将作者提供的 training graph 转为 eval graph，因为不转换为 eval 会附带很多 training 的 op， 并且有很多 op TFLite 和 Core ML 等移动框架并不支持。(最主要的问题是 TFLite 目前不支持 Bool 型标量，比如：phase_train)

```shell
python eval_graph.py model_pc model_pc_eval
```
如下所示：

![WX20190108-120019@2x.png](https://i.loli.net/2019/01/08/5c3420a7431d6.png)


使用转换后的 eval graph，将参数和结构固化，这里我们用 facenet 自带的 `freeze_graph.py` 脚本，不过由于我们之前导出的是 eval graph 所以 `phase_train` 这个参数输入被我们删除了，导致输出的 `facenet.pb` 只有一个输入节点 `input shape=(1, 64, 64, 3)` 和一个输出 `output shape=(1,512)`

```shell
python freeze_graph.py model_pc_eval facenet.pb
```

将生成的 `facenet.pb` 转化为 `tflite` 格式：

```shell
tflite_convert --output_file model_mobile_eval/facenet.tflite --graph_def_file facenet.pb  --input_arrays "input" --input_shapes "1,160,160,3" --output_arrays output --output_format TFLITE
```

祝贺你，你会在文件夹 `model_mobile_eval` 中找到 `facenet.tflite` 文件。此时如果你不再需要你的虚拟环境，你可以运行: 
```shell
pipenv rm
```
