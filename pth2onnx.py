import torch
import torch.onnx
from models import deeplabv3plus_resnet101, CWRNetPKC, CWRNet
# 设置使用CPU导出模型
device = torch.device("cpu")

def convert():
    # 模型定义来自于torchvision，样例生成的模型文件是基于resnet50模型
    model_name = "deeplab"
    model = deeplabv3plus_resnet101(n_classes=4,
                n_frames=5)


    PKC_model = torch.load('./weight/deeplabv3plus.pt', map_location='cpu')    #根据实际文件名称修改
    model.load_state_dict(PKC_model)


    if model_name == 'deeplab':
        batch_size = 1  #批处理大小
        input_shape = (5, 124, 2048)   #输入数据,改成自己的输入shape

    else:
        batch_size = 1  #批处理大小
        input_shape = (1, 5, 124, 2048)   #输入数据,改成自己的输入shape

    # 模型设置为推理模式, fastdvdnet_SNR_10.onnx 是batch size > 1, fastdvdnet_snr_10.onnx 是batch size = 1
    model.eval()

    dummy_input = torch.randn(batch_size, *input_shape) #  定义输入shape
    torch.onnx.export(model,
                      dummy_input,
                      "deeplab_model.onnx",
                      input_names = ["input"],   # 构造输入名
                      output_names = ["output"],    # 构造输出名
                      opset_version=11,    # ATC工具目前支持opset_version=9，10，11，12，13
                      dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}})  #支持输出动态轴

if __name__ == "__main__":
    convert()
