from sklearn.externals import joblib

class PcaModel:
    def __init__(self, model_male = None, model_female= None, model_joint= None):
        assert model_joint is not None or (model_male is not None and model_female is not None), 'invalid pca model configuration'
        self.model_male = model_male
        self.model_female = model_female
        self.model_joint = model_joint

    def dump(self, filepath):
        to_save = {}
        to_save['model_male'] = self.model_male
        to_save['model_female'] = self.model_female
        to_save['model_joint'] = self.model_joint
        joblib.dump(value=to_save, filename=filepath)

    def inverse_transform(self, pred):
        if len(pred.shape) == 1:
            pred = pred.reshape(1, -1)

        if self.model_joint is not None:
            verts = self.model_joint.inverse_transform(pred)
        else:
            is_male = pred[0, 0] > 0.5
            pred    = pred[:, 1:]
            if is_male:
                verts = self.model_male.inverse_transform(pred)
            else:
                verts = self.model_female.inverse_transform(pred)
        return verts

    @classmethod
    def predict(self, param):
        pass

    @staticmethod
    def load(filepath):
        data = joblib.load(filepath)
        model_male   = data['model_male']
        model_female = data['model_female']
        model_joint  = data['model_joint']
        return PcaModel(model_male=model_male, model_female=model_female, model_joint=model_joint)

