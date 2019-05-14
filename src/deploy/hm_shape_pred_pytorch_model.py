import  torch
import numpy as np
from pca.nn_util import crop_silhouette_pair
from torch.autograd import Variable

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

    def predict(self, sil_f, sil_s, height, gender):
        assert len(sil_f.shape) == 2
        assert len(sil_s.shape) == 2

        size = self.image_input_shape
        sil_f, sil_s = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1], px_height=int(0.9 * size[0]))

        sil_f = sil_f[np.newaxis, np.newaxis, :]
        sil_s = sil_s[np.newaxis, np.newaxis, :]

        aux = np.array([height])[np.newaxis, :]
        aux = self.aux_input_transform.transform(aux)
        aux = np.array([aux[0,0], gender])[np.newaxis, :]

        aux = aux.astype(np.float32)
        sil_f = sil_f.astype(np.float32)
        sil_s = sil_s.astype(np.float32)

        with torch.no_grad():
            height_var = Variable(torch.from_numpy(aux)).cuda()

            input_f_var = Variable(torch.from_numpy(sil_f)).cuda()
            input_s_var = Variable(torch.from_numpy(sil_s)).cuda()

            pred = self.model(input_f_var, input_s_var, height_var)
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