import torch
import os
from model import Classifer18

# 加载训练好的模型
model = Classifer18()
model.load_state_dict(torch.load(os.path.join('model','train0630-Res18-1','best.pth')))
model.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 512, 7, 7)  # 假设你的模型输入大小为 (1, 512, 7, 7)

# 导出为 ONNX 格式
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
