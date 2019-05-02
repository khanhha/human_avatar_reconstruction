# CNN architectures
&NewLine;
- Front/Side Architecture
&NewLine;
  <img src='https://g.gravizo.com/svg?
 digraph G {
   main[label="front/side silhouette" shape=box];
   cnn[label="CNN: Densenet"  shape=box];
   fcn[label="FCN_cnn: fully connected layers"  shape=box];
   fcn_1[label="FCN_aux: fully connected layers"  shape=box];
   fcn_2[label="FCN_final: fully connected layers"  shape=box];
   aux[label="aux_input: gender, height"  shape=box]
   concat[label="concatenation"  shape=box]
   pca[label="output: 1 gender indicator + 50 pca values"  shape=box]
   main -> cnn;
   cnn -> fcn;
   aux -> fcn_1;
   fcn -> concat;
   fcn_1 -> concat;
   concat -> fcn_2;
   fcn_2 -> pca
   }
'/>

- Joint Architecture
&NewLine;

  &nbsp;
  <img src='https://g.gravizo.com/svg?
  digraph G {
   f_sil[label="front silhouette" shape=box];
   s_sil[label="side silhouette" shape=box];
   f_cnn[label="pre-trained front CNN" shape=box];
   s_cnn[label="pre-trained side CNN" shape=box];
   fs_concat[label="concatenation" shape=box];
   fs_fcn[label="fcn: fully connected layers" shape=box];
   aux[label="aux_input: height, gender" shape=box];
   f_fcn_aux[label="fcn_aux_f: pre-trained front fcn layers" shape=box];
   s_fcn_aux[label="fcn_aux_s: pre-trained side fcn layers" shape=box];
   aux_elmwise_max[label="element-wise maximum" shape=box];
   concat_1[label="concatenation" shape=box];
   final_fcn[label="fcn_final: fully connected layers" shape=box]
   output[label="output: 1 gender indicator + 50 pca values" shape=box]
   f_sil->f_cnn;
   s_sil->s_cnn;
   f_cnn->fs_concat;
   s_cnn->fs_concat;
   fs_concat->fs_fcn;
   aux -> f_fcn_aux;
   aux -> s_fcn_aux;
   f_fcn_aux -> aux_elmwise_max;
   s_fcn_aux -> aux_elmwise_max;
   fs_fcn -> concat_1;
   aux_elmwise_max -> concat_1;
   concat_1 -> final_fcn;
   final_fcn -> output
 }
 '/>
 &NewLine;
