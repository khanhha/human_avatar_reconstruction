# notation
- caesar meshes with maxplank topology: mpii-caesar mesh
- caesar meshes with victoria topology: vic-caesar mesh

# training data preparation
- what is a victoria-based PCA model?
    - a victoria-based PCA model of 50 principal components has dimension of 50x72576x3 where 72576 is the number of victoria vertices and 3 represents 3 dimension x,y,z
    - each component of 72675x3 represents vertex devications from the mean victoria, which is averaged over all training instances.   

- PCA models training
    - the PCA model is trained using [IncrementalPCA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py) in sklearn.
   The incremental PCA model is chosen because the normal PCA model requires the whole training data to be loaded on memory, which is impossible with a PC of 16GB Ram.


|`pca models` | `training data` | `principal components` |
|:-----------:|:---------------:|:----------------------:|
| male model | 2150 vic-caesar male meshes | 50 |
|female model| 2150 vic-caesar female meshes | 50|
|joint model| 4300 vic-caesar meshes| 50 |


# CNN architectures
&NewLine;
- Front/Side Architecture

&NewLine;
    <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20main[label=%22front/side%20silhouette%22%20shape=box];%20cnn[label=%22CNN:%20Densenet%22%20shape=box];%20fcn[label=%22FCN_cnn:%20fully%20connected%20layers%22%20shape=box];%20fcn_1[label=%22FCN_aux:%20fully%20connected%20layers%22%20shape=box];%20fcn_2[label=%22FCN_final:%20fully%20connected%20layers%22%20shape=box];%20aux[label=%22aux_input:%20gender,%20height%22%20shape=box]%20concat[label=%22concatenation%22%20shape=box]%20pca[label=%22output:%201%20gender%20indicator%20+%2050%20pca%20values%22%20shape=box]%20main%20-%3E%20cnn;%20cnn%20-%3E%20fcn;%20aux%20-%3E%20fcn_1;%20fcn%20-%3E%20concat;%20fcn_1%20-%3E%20concat;%20concat%20-%3E%20fcn_2;%20fcn_2%20-%3E%20pca%20}'/>


- Joint Architecture
&NewLine;

  &nbsp;
  <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20f_sil[label=%22front%20silhouette%22%20shape=box];%20s_sil[label=%22side%20silhouette%22%20shape=box];%20f_cnn[label=%22pre-trained%20front%20CNN%22%20shape=box];%20s_cnn[label=%22pre-trained%20side%20CNN%22%20shape=box];%20fs_concat[label=%22concatenation%22%20shape=box];%20fs_fcn[label=%22fcn:%20fully%20connected%20layers%22%20shape=box];%20aux[label=%22aux_input:%20height,%20gender%22%20shape=box];%20f_fcn_aux[label=%22fcn_aux_f:%20pre-trained%20front%20fcn%20layers%22%20shape=box];%20s_fcn_aux[label=%22fcn_aux_s:%20pre-trained%20side%20fcn%20layers%22%20shape=box];%20aux_elmwise_max[label=%22element-wise%20maximum%22%20shape=box];%20concat_1[label=%22concatenation%22%20shape=box];%20final_fcn[label=%22fcn_final:%20fully%20connected%20layers%22%20shape=box]%20output[label=%22output:%201%20gender%20indicator%20+%2050%20pca%20values%22%20shape=box]%20f_sil-%3Ef_cnn;%20s_sil-%3Es_cnn;%20f_cnn-%3Efs_concat;%20s_cnn-%3Efs_concat;%20fs_concat-%3Efs_fcn;%20aux%20-%3E%20f_fcn_aux;%20aux%20-%3E%20s_fcn_aux;%20f_fcn_aux%20-%3E%20aux_elmwise_max;%20s_fcn_aux%20-%3E%20aux_elmwise_max;%20fs_fcn%20-%3E%20concat_1;%20aux_elmwise_max%20-%3E%20concat_1;%20concat_1%20-%3E%20final_fcn;%20final_fcn%20-%3E%20output%20}'/>
 &NewLine;

# Training
training consists of two main steps:
- train front model
  - input:
    - Nx384x256: N front silhouettes of size 384x256
    - Nx2: (height + gender) pairs
  - target:
    - Nx51 values: 1 gender indicator + 50 pca values

- train side model
  - input:
    - Nx384x256: N side silhouettes of size 384x256
    - Nx2: (height + gender) pairs
  - target:
    - Nx51: 1 gender indicator + 50 pca values

- train joint model based on pre-trained weights from front and side models
  - input:
    - Nx2x384x256: N front and size silhouettes of size 384x256
    - Nx2: (height+ gender) pairs
  - target:
    - NX51: 1 gender indicator + 50 pca values
