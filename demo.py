import torch
from torchvision import transforms
from BaseCNN import BaseCNN
from Main import parse_config
from Transformers import AdaptiveResize
from PIL import Image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform
test_transform = transforms.Compose([
    AdaptiveResize(768),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def main(config):

    # model
    model = BaseCNN(config)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # weight
    ckpt = './model.pt'

    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint)

    # images (path)
    path = config.path
    pathList = os.listdir(path)

    total = 0
    count = 0

    for item in pathList:
        #print(item)

        # images (single)
        image1 = Image.open(os.path.join(path, item))
        #image1 = './demo/test1.JPG'
        #image1 = Image.open(image1)
        image1 = test_transform(image1)
        image1 = torch.unsqueeze(image1, dim=0)
        image1 = image1.to(device)

        with torch.no_grad():
            score1, _ = model(image1)

        score1 = score1.cpu().item()

        total += score1
        count += 1

    print('The predicted quality is %.3f' % (total/count))


if __name__ == '__main__':
    config = parse_config()
    config.backbone = 'resnet34'
    config.representation = 'BCNN'

    main(config)


