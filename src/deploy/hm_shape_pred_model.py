from sklearn.externals import joblib
import tensorflow as tf
import numpy as np

class HmShapePredModel():
    def __init__(self, model_path, use_gpu = True):
        data = joblib.load(model_path)
        self.tf_graph_str = data['tf_graph_str']
        self.tf_graph_input_keys = data['tf_graph_input_keys']
        self.tf_graph_output_keys = data['tf_graph_output_keys']
        self.image_input_shape = data['image_input_shape']
        self.model_type = data['model_type']
        self.pca_model = data['pca_model']
        self.pca_target_transform = data['pca_target_transform']
        self.aux_input_transform =  data['aux_input_transform']

        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(self.tf_graph_str)

            elm_keys = self.tf_graph_input_keys + self.tf_graph_output_keys
            elms = tf.import_graph_def(graph_def, name='cnn_human_graph', return_elements=elm_keys)
            self.tf_graph_inputs = elms[:len(self.tf_graph_input_keys)]
            self.tf_graph_outputs = elms[len(self.tf_graph_input_keys):]

    def predict(self, sil_f, sil_s, height, gender):
        assert len(sil_f.shape) == 2
        assert len(sil_s.shape) == 2
        assert sil_f.shape == sil_s.shape
        assert sil_f.dtype == sil_s.dtype

        #size = self.image_input_shape
        #if sil_f.shape != size:
        #    sil_f, sil_s, _, _ = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1], px_height=int(0.9 * size[0]))

        if sil_f.dtype == np.uint8:
            sil_f = sil_f.astype(np.float)/255.0
            sil_s = sil_s.astype(np.float)/255.0

        sil_f = sil_f[np.newaxis, np.newaxis, :]
        sil_s = sil_s[np.newaxis, np.newaxis, :]

        aux = np.array([height])[np.newaxis, :]
        aux = self.aux_input_transform.transform(aux)
        aux = np.array([aux[0,0], gender])[np.newaxis, :]

        with self.graph.as_default():
            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                if self.model_type == 'joint':
                    feed_dict = { self.tf_graph_inputs[0] : sil_f, self.tf_graph_inputs[1] : sil_s, self.tf_graph_inputs[2] : aux }
                elif self.model_type == 'f':
                    feed_dict = { self.tf_graph_inputs[0] : sil_f, self.tf_graph_inputs[1] : aux }
                elif self.model_type == 's':
                    feed_dict = {self.tf_graph_inputs[0]: sil_s, self.tf_graph_inputs[1]: aux}
                else:
                    assert False, 'unsupported shape model type'

                preds= sess.run(self.tf_graph_outputs, feed_dict=feed_dict)
                pred = preds[0]
                pca_val = self.pca_target_transform.inverse_transform(pred)
                verts = self.pca_model.inverse_transform(pca_val)
                return verts

if __name__ == '__main__':
    path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_384_256_ml_fml/models/joint/deploy_model.jlb'
    model = HmShapePredModel(path)

    sil_f = np.zeros((384, 256), dtype=np.float)
    sil_s = np.zeros((384, 256), dtype=np.float)
    height = 1.6
    gender = 0
    pred = model.predict(sil_f, sil_s, height, gender)[0]
    pred = pred.reshape(pred.shape[0]//3, 3)
    print(f'output mesh vertex shape: {pred.shape}')