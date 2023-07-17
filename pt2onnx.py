import numpy as np
import torch
import torchvision
import onnx
import onnxruntime

from models.vit_slim_pw import ViT_slim
from utils import load_partial_weight

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    path = 'checkpoint/pruned_0.4.pth'

    cfg_prune = [[696, 2077], [732, 2047], [732, 2071], [444, 2008], [744, 2066], [756, 2120],
                 [504, 2157], [660, 2258], [720, 2420], [552, 1730], [756, 892], [696, 517]]
    # 定义一个新的模型结构复制参数
    model = ViT_slim(
        image_size=224,
        patch_size=16,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True,
        cfg=cfg_prune)  # 根据上面得到的剪枝cfg_prune重新定义网络
    model.to(device)

    weight = torch.load(path)
    load_partial_weight(model, weight)

    model.eval()
    x = torch.rand(1, 3, 224, 224, requires_grad=True).to(device)
    torch_out = model(x)
    onnx_file_name = "checkpoint/3_vit_pruned_0.4.onnx"
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_file_name,  # where to save the model (can be a file or file-like object)
                      input_names=["input"],
                      output_names=["output"],
                      verbose=False)
    # check onnx model
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
if __name__ == '__main__':
    main()

