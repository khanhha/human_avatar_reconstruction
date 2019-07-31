from face_parse.model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import cv2 as cv

class FaceParser:
    def __init__(self, model_dir):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        save_pth = osp.join(*[model_dir, "face_parsing_model.pth"])
        self.net.load_state_dict(torch.load(save_pth))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # RGB image
    def parse_face(self, img):
        resized = False
        if img.shape[0] != 512 or img.shape[1] != 512:
            img_1 = cv.resize(img, dsize=(img.shape[1], img.shape[0]), interpolation=cv.INTER_LINEAR)
            resized = True
        else:
            img_1 = img

        with torch.no_grad():
            img_1 = self.to_tensor(img_1)
            img_1 = torch.unsqueeze(img_1, 0)
            img_1 = img_1.cuda()
            out = self.net(img_1)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        if resized:
            parsing = cv.resize(parsing, dsize=(img.shape[1], img.shape[0]), interpolation=cv.INTER_NEAREST)

        return parsing