import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SMPLLoss(torch.nn.Module):

    def __init__(self, num_classes, pca_comps_male = None, pca_comps_female = None, mesh_vert_idxs = None, pca_start_idx = 1, use_weighted_mse_loss = True):
        """
        :param pca_comps_male:
        :param pca_comps_female:
        :param mesh_vert_idxs: the sparse vertex index list of the template mesh, based on which the loss will be computed; otherwise, the mesh loss
        will be computed as the MSE on all vertices. this data is only used when pca_comps_* are available.
        :param num_classes:
        :param pca:
        :param pca_weight:
        :param use_pca_loss:
        :param pca_start_idx:
        #TODO: this is a bit tricky here. normally, the target includes both gender and pca values. target[1] = gender. target[1:51] = 50 pca values;
        therefore, we have to use this pca_start_idx to mark where the pca values begin to reconstruct the mesh from it.
        :param use_weighted_mse_loss:
        """
        super(SMPLLoss, self).__init__()
        if pca_comps_male is not None and pca_comps_female is not None:
            self.use_mesh_loss = True
            self.pca_comps_male = torch.from_numpy(pca_comps_male.astype(np.float32)).cuda()
            self.pca_comps_female = torch.from_numpy(pca_comps_female.astype(np.float32)).cuda()
            if mesh_vert_idxs is not None:
                self.mesh_vert_idxs = torch.from_numpy(mesh_vert_idxs).cuda()
            else:
                self.mesh_vert_idxs = None
            self.pca_start_idx = pca_start_idx
        else:
            self.use_mesh_loss = False
            self.pca_comps_male = None
            self.pca_comps_female = None
            self.mesh_vert_idxs = None
            self.pca_start_idx = None

        self.mesh_loss_weight = 0.0

        self.use_weighted_mse_loss = use_weighted_mse_loss
        if use_weighted_mse_loss:
            mse_w = torch.ones(num_classes)
            mse_w[0] = 10
            mse_w[1] = 10
            mse_w[2] = 9
            mse_w[3] = 8
            mse_w[4] = 7
            mse_w[5] = 6
            mse_w[6] = 5
            mse_w[7] = 4
            mse_w[8:10] = 3
            mse_w[10:20] = 2
            mse_w[20:] = 1
            #TODO: if we don't normalize weight this way, the reuslt looks worse. WHY?
            mse_w = mse_w / mse_w.sum()
            #print(mse_w)
            self.mse_w = mse_w.cuda()
            self.mse = torch.nn.MSELoss(reduce=False)
        else:
            self.mse = torch.nn.MSELoss()

    def mse_loss(self, y_pred, y):
        if self.use_weighted_mse_loss:
            loss = self.mse(y_pred, y)
            loss = loss * self.mse_w
            return loss.mean()
        else:
            return self.mse(y_pred, y)

    def loss_update_per_epoch(self, epoch):
        start = 0.05
        #decrease decay rate
        d = (epoch + 1) / 2
        mse_loss_weight = min(1.0, start + (1.0-start)**d)
        self.mesh_loss_weight = 1.0 - mse_loss_weight

    def forward(self, pred, target):
        if self.use_weighted_mse_loss:
            mse_lss = self.mse(pred, target)
            mse_lss = (mse_lss * self.mse_w).mean()
        else:
            mse_lss = self.mse(pred, target)

        if self.use_mesh_loss:
            #TODO: the mesh loss in general is at a smaller sclae than the PCA loss.
            # the range of PCA values could be from [-3, 3]
            # while maximum vertex difference are just within range of [-0.1, 0, 0.1], I guess. This is becaus the height of mesh is from [0, 1.6], so the mean vertex pairwise distance
            # maybe just around 0.1
            #pca_mode.inverse_transform.  see _BasePCA.inverse_transform for more detail
            if self.mesh_vert_idxs is None:
                #difference of all mesh vertices
                mesh_lss_male = torch.mm(pred[:, self.pca_start_idx:], self.pca_comps_male) - torch.mm(target[:, self.pca_start_idx:], self.pca_comps_male)
            else:
                #differece of a subset of vertex mesh
                m_pred_male = torch.mm(pred[:, self.pca_start_idx:], self.pca_comps_male)
                m_pred_male = m_pred_male.view(m_pred_male.shape[0], -1, 3)
                #select a subset of vertices
                m_pred_male_subset = m_pred_male[:, self.mesh_vert_idxs, :]

                m_y_male = torch.mm(target[:, self.pca_start_idx:], self.pca_comps_male)
                m_y_male = m_y_male.view(m_y_male.shape[0], -1, 3)
                #select a subset of vertices
                m_y_male_subset = m_y_male[:, self.mesh_vert_idxs, :]

                #calculate the vertex differences
                mesh_lss_male = (m_pred_male_subset - m_y_male_subset).view(m_y_male_subset.shape[0], -1)

            male_model_indicator = target[:,0]
            mesh_lss_male_per_mesh = torch.mean(mesh_lss_male**2, dim=1) * male_model_indicator
            mesh_lss_male_per_batch = torch.mean(mesh_lss_male_per_mesh)

            #pca_mode.inverse_transform.  see _BasePCA.inverse_transform for more detail
            if self.mesh_vert_idxs is None:
                #difference of all mesh vertices
                mesh_lss_female = torch.mm(pred[:, self.pca_start_idx:], self.pca_comps_female) - torch.mm(target[:, self.pca_start_idx:], self.pca_comps_female)
            else:
                #differece of a subset of vertex mesh
                m_pred_female = torch.mm(pred[:, self.pca_start_idx:], self.pca_comps_female)
                m_pred_female = m_pred_female.view(m_pred_female.shape[0], -1, 3)
                #select a subset of vertices
                m_pred_female_subset = m_pred_female[:, self.mesh_vert_idxs, :]

                m_y_female = torch.mm(target[:, self.pca_start_idx:], self.pca_comps_female)
                m_y_female = m_y_female.view(m_y_female.shape[0], -1, 3)
                #select a subset of vertices
                m_y_female_subset = m_y_female[:, self.mesh_vert_idxs, :]

                #calculate the vertex differences
                mesh_lss_female = (m_pred_female_subset - m_y_female_subset).view(m_y_female_subset.shape[0], -1)

            mesh_lss_female_per_mesh = torch.mean(mesh_lss_female**2, dim=1)*(1.0-male_model_indicator)
            mesh_lss_female_per_batch = torch.mean(mesh_lss_female_per_mesh)

            mesh_loss = mesh_lss_male_per_batch + mesh_lss_female_per_batch
            return (1.0 - self.mesh_loss_weight) * mse_lss + self.mesh_loss_weight * mesh_loss
        else:
            return mse_lss

from sklearn.decomposition import IncrementalPCA
import numpy as np
if __name__ == '__main__':
    batch_size = 2
    out_size = 51
    N_verts = 49963
    model = nn.Linear(20,out_size).cuda()
    x = torch.randn(batch_size, 20).cuda()
    y = torch.randn(batch_size, out_size).cuda()

    male_pca = IncrementalPCA(n_components=50)
    female_pca = IncrementalPCA(n_components=50)

    male_pca.fit(np.random.rand(200, N_verts*3).astype(np.float32))
    female_pca.fit(np.random.rand(200, N_verts*3).astype(np.float32))

    N_loss_verts = 550
    #mesh_loss_vert_idxs = torch.randint(0, N_verts, (N_loss_verts, 1)).view(-1)
    mesh_loss_vert_idxs = np.random.randint(0, N_verts, N_loss_verts)
    cri = SMPLLoss(num_classes = out_size, pca_comps_female=female_pca.components_, pca_comps_male=male_pca.components_,
                   pca_start_idx=1,
                   use_weighted_mse_loss=True, mesh_vert_idxs=mesh_loss_vert_idxs)
    pred_y = model(x)
    loss = cri(pred_y, y)
    print(loss)
    #for i in range(50):
    #    cri.loss_update_per_epoch(i)
    #    print(cri.mesh_loss_weight)

    loss.backward()