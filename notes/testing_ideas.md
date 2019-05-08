# testing in the training time
- test data:
  - make human objects, Caesar meshes in the test test (not the training set)
- test method
  - calculate measurement from the test meshes based on the fixed topology of the meshes.
  - compare MSE (mean square error) between the ground-truth measurement and measurement extracted on the predicted meshes.

# testing in the test time
  - in this case, we need ground-truth measurement from the test subjects.
  - if we don't have many subjects, we can take pictures of one subject under different background and camera conditions.

# about the Oh's suggestion to  project the predicted mesh to silhouettes and then compare silhouettes
  - we can only do it if we have projection matrix. Guessing an orthographic projection doesn't work.
