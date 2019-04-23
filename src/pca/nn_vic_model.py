import torch

class NNModelWrapper:
    def __init__(self, model, model_type, pca_model, use_pca_loss, use_height = True, pca_target_transform = None, height_transform = None):
        assert model_type in ['f', 's', 'joint']
        self.model = model
        self.model_type = model_type
        self.pca_target_transform  = pca_target_transform
        self.height_transform = height_transform
        self.use_pca_loss = use_pca_loss
        self.use_height = use_height
        self.pca_model = pca_model

    def dump(self, filepath):
        to_save = {}
        to_save['model'] = self.model
        to_save['model_type'] = self.model_type
        to_save['pca_model'] = self.pca_model
        to_save['pca_target_transform'] = self.pca_target_transform
        to_save['height_transform'] = self.height_transform
        to_save['use_pca_loss'] = self.use_pca_loss
        to_save['use_height'] = self.use_height
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

        return NNModelWrapper(model=model, model_type=model_type, pca_model=pca_model,
                              use_pca_loss=use_pca_loss, use_height=use_height,
                              pca_target_transform=pca_target_transform, height_transform=height_transform)