import torch
from igray.model import InvertibleGrayscale
from igray.dataset import pil_loader, DEFAULT_TRANSFORM
import torchvision.transforms as transforms


model = torch.load('igray_checkpoint_20190528083437.pth.tar', map_location='cpu')
state_dict = model['state_dict']

ig_net = InvertibleGrayscale()
ig_net.load_state_dict(state_dict)

X = pil_loader('/Users/a12201/Pictures/otacollection.jpg')
X = DEFAULT_TRANSFORM(X)
Y_gray, Y_restored = ig_net(X.unsqueeze(0))
Y_gray, Y_restored = (Y_gray + 1) / 2, (Y_restored + 1) / 2
Y_gray, Y_restored = transforms.ToPILImage()(Y_gray.squeeze(0)), transforms.ToPILImage(mode='RGB')(Y_restored.squeeze(0))
Y_gray.save('gray.png')
Y_restored.save('restored.png')
