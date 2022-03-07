import torch
import torchvision
from torchvision import transforms
import numpy as np

class Resnet18():
    def __init__(self) -> None:
        self.m = torchvision.models.resnet18(pretrained=True)
        self.layer4 = torchvision.models._utils.IntermediateLayerGetter(self.m,{'layer1': 'feat2'})
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Input (1, 3, 224, 224)
    # Output (1,256,14,14)
    @torch.no_grad()
    def get_feature(self,image_list,target):
        image_list = [self.preprocess(image).unsqueeze(0) for image in image_list]
        image = torch.cat(image_list,0)
        out = self.layer4(image)['feat2']
        out = torch.chunk(out,out.shape[0],0)
        out = [torch.squeeze(image) for image in out]
        RFC = self.RFC_feature(out,target)
        RFC = np.array(RFC)
        return RFC

    @torch.no_grad()
    def get_frame_feature(self,image_list):
        image_list = [self.preprocess(image).unsqueeze(0) for image in image_list]
        image = torch.cat(image_list,0)
        out = self.layer4(image)['feat2']
        out = torch.chunk(out,out.shape[0],0)
        out = [torch.squeeze(image) for image in out]
        return out

    @torch.no_grad()
    def get_feature2(self,image_list,frame=None):
        image_list = [self.preprocess(image).unsqueeze(0) for image in image_list]
        image = torch.cat(image_list,0)
        out = self.m(image)
        out = torch.chunk(out,out.shape[0],0)
        out = [torch.squeeze(image).numpy() for image in out]
        return np.array(out)

    @torch.no_grad()
    def get_frame_feature2(self,image_list):
        image_list = [self.preprocess(image).unsqueeze(0) for image in image_list]
        image = torch.cat(image_list,0)
        out = self.m(image)
        return out

    def RFC_feature(self,image_list,target):
        distance = [torch.sub(image,target).pow(2).sqrt() for image in image_list]
        return [torch.sum(t,(1,2)).numpy() for t in distance]

    def RFC_feature2(self,image_list,target):
        distance = [torch.sum(torch.sub(image,target).pow(2),(1,2)).sqrt().numpy() for image in image_list]
        return distance
        