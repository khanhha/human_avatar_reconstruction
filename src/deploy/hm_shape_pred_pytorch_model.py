import  torch
import numpy as np
from pca.nn_util import crop_silhouette_pair
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import PIL

class HmShapePredPytorchModel():
    def __init__(self, model_path):
        data = torch.load(model_path)
        self.model = data['model']
        self.model = self.model.to('cuda')
        self.model.eval()

        self.image_input_shape = (384, 256)
        self.model_type = data['model_type']
        self.pca_model = data['pca_model']
        self.pca_target_transform = data['pca_target_transform']
        self.aux_input_transform =  data['height_transform']
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        #overide default image transform if one is provided
        if 'img_input_transform' in data.keys():
            self.img_transform = data['img_input_transform']

    def predict(self, sil_f, sil_s, height, gender):
        assert len(sil_f.shape) == 2
        assert len(sil_s.shape) == 2

        if sil_f.dtype == np.uint8:
            sil_f = sil_f.astype(np.float)/255.0
        if sil_s.dtype == np.uint8:
            sil_s = sil_s.astype(np.float)/255.0

        #add batch and channel dimensions
        sil_f = sil_f[np.newaxis, :]
        sil_s = sil_s[np.newaxis, :]

        aux = np.array([height])[np.newaxis, :]
        aux = self.aux_input_transform.transform(aux)
        aux = np.array([aux[0,0], gender])[np.newaxis, :]

        aux = aux.astype(np.float32)
        sil_f = sil_f.astype(np.float32)
        sil_s = sil_s.astype(np.float32)
        sil_f = transforms.ToPILImage()(torch.tensor(sil_f))
        sil_s = transforms.ToPILImage()(torch.tensor(sil_s))
        sil_f = self.img_transform(sil_f).unsqueeze(0)
        sil_s = self.img_transform(sil_s).unsqueeze(0)

        with torch.no_grad():
            aux_var = Variable(torch.from_numpy(aux)).cuda()

            if self.model_type == 'f':
                input_f_var = Variable(sil_f).cuda()
                pred = self.model(input_f_var, aux_var)
            elif self.model_type == 's':
                input_s_var = Variable(sil_s).cuda()
                pred = self.model(input_s_var, aux_var)
            elif self.model_type == 'joint':
                input_f_var = Variable(sil_f).cuda()
                input_s_var = Variable(sil_s).cuda()
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