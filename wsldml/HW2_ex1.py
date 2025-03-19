import torch
import torch_directml
dml = torch_directml.device()
#torch-directml 的当前版本映射到“PrivateUse1”Torch 后端。 torch_directml.device() API 是一个方便的包装器，用于将张量发送到 #DirectML 设备。

#创建 DirectML 设备后，现在可以定义两个简单的张量：一个张量包含 1，另一个张量包含 2。 将张量放在“dml”设备上。



tensor1 = torch.tensor([1]).to(dml) # Note that dml is a variable, not a string!
tensor2 = torch.tensor([2]).to(dml)
#将张量相加，然后打印结果。



dml_algebra = tensor1 + tensor2
print(dml_algebra.item())