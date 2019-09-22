import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NNHmModel(nn.Module):
    def __init__(self, num_classes, n_aux_input_feature, encoder_type='densenet'):
        super(NNHmModel, self).__init__()
        if encoder_type == 'densenet':
            self.encoder =  torchvision.models.densenet121(pretrained=True)
            self.encoder_output_size = self.encoder.classifier.out_features
        elif encoder_type == 'vgg16_bn':
            self.encoder = torchvision.models.vgg16_bn(pretrained=True)
            self.encoder_output_size = self.encoder.classifier[-1].out_features
        elif encoder_type == 'resnet18':
            self.encoder = torchvision.models.resnet18(pretrained=True)
            self.encoder_output_size = self.encoder.fc.out_features
        else:
            assert False, 'Unsupported encoder'

        self.n_aux_input_feature = n_aux_input_feature
        self.n_aux_output_feature = 32
        self.aux_embedding = nn.Sequential(nn.Linear(self.n_aux_input_feature, 16),
                                          nn.ReLU(),
                                          nn.Linear(16, self.n_aux_output_feature),
                                          nn.ReLU())

        self.fusion_input_size = self.n_aux_output_feature + self.encoder_output_size
        self.fusion_output_size = 512
        self.fusion = nn.Sequential(nn.Linear(self.fusion_input_size, self.fusion_output_size),
                                    nn.ReLU(),
                                    nn.Linear(self.fusion_output_size, self.fusion_output_size),
                                    nn.ReLU())

        self.regressor = nn.Linear(self.fusion_output_size, num_classes)

    def forward(self, x, aux):
        encoder_feat = self.encoder(x)
        aux_feat = self.aux_embedding(aux)

        in_fusion_feat = torch.cat((encoder_feat, aux_feat), dim=1)
        fusion_feat = self.fusion(in_fusion_feat)

        out = self.regressor(fusion_feat)

        return out

class NNHmJointModel(nn.Module):
    def __init__(self, model_f, model_s, num_classes, fine_tune_f_s_models = False):
        super(NNHmJointModel, self).__init__()

        self.merge_method = 'max' #concat
        assert model_f.fusion_input_size == model_s.fusion_input_size, 'unmatched model_f and model_s'
        assert model_f.fusion_output_size == model_s.fusion_output_size, 'unmatched model_f and model_s'

        self.aux_f = model_f.aux_embedding
        self.encoder_f = model_f.encoder
        self.fusion_f = model_f.fusion
        self.set_parameter_requires_grad(self.aux_f, fine_tune_f_s_models)
        self.set_parameter_requires_grad(self.encoder_f, fine_tune_f_s_models)
        self.set_parameter_requires_grad(self.fusion_f, fine_tune_f_s_models)


        self.aux_s = model_s.aux_embedding
        self.encoder_s = model_s.encoder
        self.fusion_s = model_s.fusion
        self.set_parameter_requires_grad(self.aux_s, fine_tune_f_s_models)
        self.set_parameter_requires_grad(self.encoder_s, fine_tune_f_s_models)
        self.set_parameter_requires_grad(self.fusion_s, fine_tune_f_s_models)

        if self.merge_method == 'concat':
            self.input_regressor_size = model_f.fusion_output_size + model_s.fusion_output_size
        elif self.merge_method == 'max':
            self.input_regressor_size = model_f.fusion_output_size
        else:
            assert False, 'unsupported merge method'

        self.regressor = nn.Sequential(nn.Linear(self.input_regressor_size, self.input_regressor_size),
                                    nn.ReLU(),
                                    nn.Linear(self.input_regressor_size, num_classes))

    def set_parameter_requires_grad(self, model, fine_tune):
        if  not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, sil_f, sil_s, aux):
        #replicate the forward step in the front single model
        encoder_feat_f = self.encoder_f(sil_f)
        aux_feat_f = self.aux_f(aux)
        in_fusion_feat_f = torch.cat((encoder_feat_f, aux_feat_f), dim=1)
        fusion_feat_f = self.fusion_f(in_fusion_feat_f)

        #replicate the foward step of the side single model
        encoder_feat_s = self.encoder_s(sil_s)
        aux_feat_s = self.aux_s(aux)
        in_fusion_feat_s = torch.cat((encoder_feat_s, aux_feat_s), dim=1)
        fusion_feat_s = self.fusion_s(in_fusion_feat_s)

        if self.merge_method == 'max':
            fusion = torch.max(fusion_feat_f, fusion_feat_s)
        elif self.max == 'concat':
            fusion = torch.cat((fusion_feat_f, fusion_feat_s), dim=1)
        else:
            assert False, 'unsupported merge method'

        pred = self.regressor(fusion)

        return pred

class NNModelWrapper:
    def __init__(self, model, model_type, pca_model, use_pca_loss, use_height = True, img_input_transform = None, pca_target_transform = None, height_transform = None):
        assert model_type in ['f', 's', 'joint']
        self.model = model
        self.model_type = model_type
        self.pca_target_transform  = pca_target_transform
        self.height_transform = height_transform
        self.use_pca_loss = use_pca_loss
        self.use_height = use_height
        self.pca_model = pca_model
        self.img_input_transform = img_input_transform

    def dump(self, filepath):
        to_save = {}
        to_save['model'] = self.model
        to_save['model_type'] = self.model_type
        to_save['pca_model'] = self.pca_model
        to_save['pca_target_transform'] = self.pca_target_transform
        to_save['height_transform'] = self.height_transform
        to_save['use_pca_loss'] = self.use_pca_loss
        to_save['use_height'] = self.use_height
        to_save['img_input_transform'] = self.img_input_transform
        torch.save(obj=to_save, f=filepath)

    @staticmethod
    def load(filepath):
        data = torch.load(filepath)
        model = data['model']
        model_type = data['model_type']
        pca_model = data['pca_model']
        pca_target_transform = data['pca_target_transform']
        height_transform = data['height_transform']
        use_pca_loss = data['use_pca_loss']
        use_height = data['use_height']

        #backward compability
        if 'img_input_transform' in data:
            img_input_transform = data['img_input_transform']
        else:
            img_input_transform = None

        return NNModelWrapper(model=model, model_type=model_type, pca_model=pca_model,
                              use_pca_loss=use_pca_loss, use_height=use_height,
                              img_input_transform=img_input_transform, pca_target_transform=pca_target_transform, height_transform=height_transform)

import numpy as np
from pca.losses import SMPLLoss

if __name__ == '__main__':
    encoder_type = 'resnet18'
    model_f = NNHmModel(encoder_type=encoder_type, num_classes=51, n_aux_input_feature=2)
    mode_f = model_f.cuda()
    model_s = NNHmModel(encoder_type=encoder_type, num_classes=51, n_aux_input_feature=2)
    mode_s = model_s.cuda()

    sil_f   = torch.tensor(np.random.rand(4, 3, 224, 224).astype(np.float32)).cuda()
    sil_s   = torch.tensor(np.random.rand(4, 3, 224, 224).astype(np.float32)).cuda()
    aux = torch.tensor(np.random.rand(4,2).astype(np.float32)).cuda()

    pred_f = model_f(sil_f, aux)
    pred_s = model_s(sil_s, aux)
    ##print(pred.data.cpu().numpy())

    model_joint = NNHmJointModel(model_f, model_s, 51, fine_tune_f_s_models=False).cuda()
    pred_1 = model_joint(sil_f, sil_s, aux)
    print(pred_1.data.cpu().numpy().shape)

