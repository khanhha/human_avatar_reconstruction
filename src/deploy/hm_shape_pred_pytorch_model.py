import  torch
import numpy as np
from pca.nn_util import crop_silhouette_pair
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import PIL
import time

class HmShapePredPytorchModel():
    def __init__(self, model_path, use_gpu = True):
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        data = torch.load(model_path, map_location=self.device)

        self.model = data['model']
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_type = data['model_type']
        self.pca_model = data['pca_model']
        self.pca_target_transform = data['pca_target_transform']
        self.aux_input_transform =  data['height_transform']
        #overide default image transform if one is provided
        if 'img_input_transform' in data.keys():
            self.img_transform = data['img_input_transform']
            print('shape model: image transformation: ', self.img_transform)
        else:
            self.img_transform = transforms.Compose([transforms.ToTensor()])
            print("shape mode: use default input image transformation")

    def predict(self, img_f, img_s, height, gender):
        if len(img_f.shape) == 2:
            #TODO: very complex and bad transformation
            if img_f.dtype == np.uint8:
                img_f = img_f.astype(np.float) / 255.0
            if img_s.dtype == np.uint8:
                img_s = img_s.astype(np.float) / 255.0

            #add batch and channel dimensions
            img_f = img_f[np.newaxis, :]
            img_s = img_s[np.newaxis, :]

            img_f = img_f.astype(np.float32)
            img_s = img_s.astype(np.float32)
            img_f = transforms.ToPILImage()(torch.tensor(img_f))
            img_s = transforms.ToPILImage()(torch.tensor(img_s))
            img_f = self.img_transform(img_f).unsqueeze(0)
            img_s = self.img_transform(img_s).unsqueeze(0)
        else:
            img_f = transforms.ToPILImage()(img_f)
            img_s = transforms.ToPILImage()(img_s)
            img_f = self.img_transform(img_f).unsqueeze(0)
            img_s = self.img_transform(img_s).unsqueeze(0)

        aux = np.array([height])[np.newaxis, :]
        aux = self.aux_input_transform.transform(aux)
        aux = np.array([aux[0,0], gender])[np.newaxis, :]
        aux = aux.astype(np.float32)

        with torch.no_grad():
            aux_var = Variable(torch.from_numpy(aux)).to(self.device)

            if self.model_type == 'f':
                input_f_var = Variable(img_f).to(self.device)
                pred = self.model(input_f_var, aux_var)
            elif self.model_type == 's':
                input_s_var = Variable(img_s).to(self.device)
                pred = self.model(input_s_var, aux_var)
            elif self.model_type == 'joint':
                input_f_var = Variable(img_f).to(self.device)
                input_s_var = Variable(img_s).to(self.device)
                pred = self.model(input_f_var, input_s_var, aux_var)
            pred = pred.data.cpu().numpy()

            pred = self.pca_target_transform.inverse_transform(pred.reshape(1,-1))
            verts = self.pca_model.inverse_transform(pred)

            return verts

if __name__ == '__main__':
    #path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml/models/joint/deploy_model.jlb'
    path =  "/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models/shape_model_pytorch.pt"
    model = HmShapePredPytorchModel(path)

    sil_f = np.zeros((384, 256), dtype=np.float)
    sil_s = np.zeros((384, 256), dtype=np.float)
    height = 1.6
    gender = 0
    pred = model.predict(sil_f, sil_s, height, gender)
    print(f'output mesh vertex shape: {pred.shape}')