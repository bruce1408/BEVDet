import os
import os 
import shutil 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
import torchvision.models as models 
from tqdm import tqdm
import argparse
import io 
import torch 
import onnx

from nquantizer import run_quantizer 
from ncompiler import run_compiler



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18), resnet34, resnet50, mobilenetv2, efficientnet-b3')

args = parser.parse_args()

# model = models.__dict__[args.arch](pretrained=True)


# case_name = "resnet18_imagenet"

data_dir = "/opt/datasets/imagenet/"

work_dir = "output/" + args.arch

# model_file = 'resnet18-5c106cde.pth'


if os.path.exists(work_dir):
    shutil.rmtree(work_dir) 
os.makedirs(work_dir, exist_ok=True)

input_shape = (1, 3, 224, 224)


def get_model():
    model = models.__dict__[args.arch](pretrained=True)
    # model = models.resnet18(pretrained=True) 
    # state_dict = torch.load(model_file) 
    # model.load_state_dict(state_dict) 
    return model


cls_list = ['n01930112', 'n02110341', 'n04067472', 'n04509417']

synset = {l.strip()[:9]: l.strip()[9:] for l in open('/opt/datasets/synset.txt').readlines()} 
cls_index = [list(synset.keys()).index(cls) for cls in cls_list]
print('cls index is : ', len(cls_index))

def get_dataloader():
    val_dir = data_dir
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
    ),
    batch_size = 1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    )
    return val_loader


def get_metric(model):
    val_loader = get_dataloader()
    correct_1 = 0 
    correct_5 = 0 
    total = 0 
    for data in tqdm(val_loader):

        images, labels = data

        labels = torch.tensor(cls_index)[labels]

        images = images.cuda() 
        labels = labels.cuda() 
        output = model(images)
        _, predict = output.topk(1, 1, True, True) 
        predict = predict.t() 
        correct = predict.eq(labels.view(1, -1).expand_as(predict)) 
        correct_1 += correct[:1].flatten().float().sum(0, keepdim=True) 
        correct_5 += correct[:5].flatten().float().sum(0, keepdim=True) 
        total += labels.shape[0] 
    return float(correct_1) / float(total) * 100


model = get_model().eval()

buffer = io.BytesIO()

torch.onnx.export(model, torch.randn(input_shape), buffer, opset_version=11)
onnx_model = onnx.load_from_string(buffer.getvalue())
val_loader = get_dataloader()
input_vars = [torch.randn(input_shape).cuda()]


# run_quantizer接⼝说明请参考NPU⼯具链开发⼿册量化⼯具章节 
# 
quant_model = run_quantizer(
    onnx_model, 
    dataloader=val_loader, num_batches=200,
    output_dir=work_dir + "/quantizer_output",
    input_vars=input_vars,
)

float_metric = get_metric(model.cuda())
quant_metric = get_metric(quant_model.cuda())
print("float_metric:", float_metric, " quant_metric:", quant_metric)


# run_compiler接⼝说明请参考NPU⼯具链开发⼿册编译器章节 
run_compiler(
    input_dir=work_dir + "/quantizer_output",
    output_dir=work_dir + "/compiler_output", 
    enable_simulator=True, 
    enable_profiler=True,
)