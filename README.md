
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Web portal](#web-portal)
- [Quick inference](#quick-inference)
- [Training](#training)
- [documentation index](#documentation-index)
  - [body reconstruction](#body-reconstruction)
  - [head reconstruction](#head-reconstruction)
  - [head reconstruction](#head-reconstruction)
<!-- /code_chunk_output -->

# Web portal
The web portal demo contains features
- manage and store multiple subjects in a SQL database
- predict 3D human shapes for subject
- calculate measurements on the predicted 3D body shapes.
- support compare the measurement error given the groundtruth measurements.

To bring up the web portal with the pre-trained models (stored in google drive), please follows [the instructions](/notes/web_portal_instruction.md)

# Quick inference
[The instructions](notes/cnn_pipeline_instruction.md)

# Training
To bring up the training, please follow [the instructions](notes/cnn_pipeline_instruction.md)

# documentation index
## body reconstruction
- [explanation of stages in shape model training](./notes/cnn_pipeline.md)
- [improvement ideas for training the shape model ](./notes/cnn_improvement_list.md)
- [the instructions of transferring shapes to Victoria](./notes/vic_mpii_deformation_pipeline.md)
- [a summary of the effect of camera properties on silhouette](./notes/cnn_camera_effect.md)
- [testing ideas for the cnn-based method](notes/testing_ideas.md)
- [a summary of the slice-based method](./notes/slice_method_summary.md)
- [victoria-caesar deformatin pipeline](./notes/vic_mpii_deformation_pipeline.md)

## head reconstruction
- [explanation of the head reconstruction pipelin](./notes/head_reconstruction.md)
- [head reconstruction paper list](./notes/head_reconstruction_paper_list.md)
