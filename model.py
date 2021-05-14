import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from importlib import import_module, reload





# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, args=None):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class FasterrcnnResnet50Fpn(nn.Module):
    def __init__(self, num_classes=11, args=None):
        super().__init__()
        
        # fix refenence: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # load model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # set box predictor layer
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        

    def forward(self, *x):
        
        return self.model(*x)
    
    
class FasterRcnnCustom(nn.Module):
    def __init__(self, num_classes=11, args=None):
        super().__init__()
        
        backbone_module = getattr(import_module("torchvision.models"), args.backbone)
        backbone = backbone_module(pretrained=True).features
        backbone.out_channels = backbone(torch.zeros(1,3,512,512)).shape[1]
                
        anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))


        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

        self.model = torchvision.models.detection.FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)




        
        
    def forward(self, *x):
        return self.model(*x)
    
    
