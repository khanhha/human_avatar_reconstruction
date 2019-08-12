
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Pipeline](#pipeline)
  - [Run PRN facelib.](#run-prn-facelib)
  - [Adjust input image (optional stage)](#adjust-input-image-optional-stage)
    - [Idea 1](#idea-1)
    - [Idea 2](#idea-2)
    - [Problems with seamless cloning.](#problems-with-seamless-cloning)
  - [Texuture mapping](#texuture-mapping)
- [Problems](#problems)
  - [Position map outisde factial region causes backgound in the final texture](#position-map-outisde-factial-region-causes-backgound-in-the-final-texture)
    - [Experiment 1: make background black in a hope to increase PRN facelib accurarcy](#experiment-1-make-background-black-in-a-hope-to-increase-prn-facelib-accurarcy)
  - [Incorrect nostril color problem](#incorrect-nostril-color-problem)
    - [Current state](#current-state)
    - [Problem description](#problem-description)
  - [PRN facelib training data quality.](#prn-facelib-training-data-quality)
- [Improvement ideas](#improvement-ideas)
  - [Skin synthesis](#skin-synthesis)

<!-- /code_chunk_output -->



# Pipeline
## Run PRN facelib.
The result consists of the following information
- Position map: a 3-channel image of size (256x256) that contains (x,y,z) coordinates of the 3D face.
This position map is the direct output of the convolution neural network of the PRN facelib.
From this (x,y,z) position map, we can infer the depth map, texture map and landmark positions.
Below is the plotting of the (x,y) channels of the position map over the input image.

![](images/.skin_texture_pipeline_images/pos_map.png)

- 68 facial landmarks: these landmarks are extracted directly from the position map image using
a pre-defined set of x,y indices to the position map. Note that these x,y values are different from the (x,y,z) values stored in the position map. Below are 68 landmarks plotted on the face image. Due to the position map is incorrect, the landmarks are incorrect as well, as shown by the red points inside the green contour.

![](images/.skin_texture_pipeline_images/landmarks.png)

- Texture map: It is made of the (x,y) channels of the position map.
Specifically, it is a mapping from the facial region of the input image to a rectangular texture of (256x256). Below is the visuaslization of a texture extracted using this texture map. The black/white area along jaw is due tu the inaccuracy of the position map in that region.

![](images/.skin_texture_pipeline_images/texture_map.png)

## Adjust input image (optional stage)
The texture map (x,y channels of position map) maps (R,G,B) pixels from the facial region of the input image to the 256x256 texture map. There are two problems regardign this mapping.
- Because the position map is incorrect, it also maps some background area along jaw from the input image to the texture, as explained in the previous stage.
- The texture map also maps hair to the texture.

![](images/.skin_texture_pipeline_images/hair.png)

### Idea 1
- Apply grabcut to detect skin region. The foreground mask for grabcut is created by exclude the eye and lip polygons from the convex polygon of the 68 landmarks, and then erode with a kernel of 10x10 to make sure that there is no wall/background color inside the foreground mask.

![](images/.skin_texture_pipeline_images/grabcut.png)

- Create a background image filled with the estimated skin color.
- Apply seamless cloning to transfer the facial area with grab-cut mask to the skin background image. The middle figure shows result without cloning, just replace every pixels outside the skin mask by estimated color. The right-most figure shows the result with seamless cloning.

![](images/.skin_texture_pipeline_images/seamless_cloning.png)

### Idea 2
One of the two main motivations behind skin segmentation is that the (x,y,y) position map from the PRN facelib covers some background region, which bring background pixels into the final texture. There are two other approaches to solve this problem.
- Because the error just often occur along the jaw area, we can pre-define a mask over the Victoria texture map that specifies possible error regions so that we can replace them by skin color later. This approach is very simple. It could be very effective if we could have a good skin color estimation and a smooth seam between the real facial region and skin color regions.
- Another approach is that we can post-process the (x,y,z) position map by warping them in a a way that (x,y) locations will stay inside the facial region. To do this, we can use OpenCV to detect another 68 facial landmarks to be used as target for warping algorithm. However, there could be two further problems with this approach. Firstly, we are also not sure about how good the OpenCV/Dlib landmarks are. Warping based on just facial landmarks might not enough because some points in the (x,y,z) position maps lie outside the polygon formed by the facial landmarks. Therefore, even that facial landmarks are well aligned with facial contours, the position map could still cover some background regions.

### Problems with seamless cloning.
-  Seamless cloning modifies the foreground object color (in our case, the foreground is the facial region) to match to the background color. Therefore, when the background color (estimated skin color) is too far away from the face color, it will change the face color. As an example, you can see the color of the lion in the below figure is transformed toward green to match the background color. For more examples about seamless cloning, check [this link](http://www.ctralie.com/Teaching/PoissonImageEditing/)

![](images/.skin_texture_pipeline_images/seamless_clonign_exp.png)
- In some cases, even that the estimated skin color is quite similar to the face color, seamless cloning saturates the face region toward white. In my opinion, it could be possible that in these cases, both estimated skin and face colors are already too bright. When I tried to reduce the brightness of the estimated skin color (skin_color = 0.85*skin_clor), the seamless cloning result becomes less saturated.  

![](images/.skin_texture_pipeline_images/seamless_cloning_saturated.png)


## Texuture mapping
- step 1: Map the modified input image to PRN facelib texture

![](images/.skin_texture_pipeline_images/input_img_to_prn_tex.png)

- step 2: Map the PRN facelib texture to the Victoria texture space. the yellow mask is a predefined texture mask that specifies our interest region inside the PRN texture space. We will use seamless cloning again here to clone this yellow region to the Victoria texture space

![](images/.skin_texture_pipeline_images/prn_tex_to_vic_tex.png)


# Problems
## Position map outisde factial region causes backgound in the final texture
### Experiment 1: make background black in a hope to increase PRN facelib accurarcy
Result: prediction result is still similar. Background still interferes with texture

![](images/.skin_texture_pipeline_images/f0ebd802.png)

## Incorrect nostril color problem

### Current state
The problem is solved quite well so far by applying inpainting in the nostril region.
![](images/.skin_texture_pipeline_images/b48bd75f.png)

### Problem description
The below picture visualizes the black artifact under nostril. Specifically, the bottom of nostril
has black color but the nostril itself is kind of brighter.

![](images/.skin_texture_pipeline_images/0bc8c8b0.png)

The below figure visualize the mapping/warping from the input image to the PRN texture.

The blue points in the left image denotes the (x,y) coordinates of 3D points in the position map from PRN facelib. In other words, these blue points represent an orthogonal projection of vertices of the 3D face mesh (the position map) onto the input image face; therefore, one blue point in the left image could be the projections of multiple 3D points in the 3D face mesh.

The right image is the PRN facelib texture. Because this is a texture, each of its pixels is mapped uniquely to one 3D point in the 3D mesh. Also, each of its pixel is mapped to one blue point in the left image; however, this could be many-to-one mapping because one blue point could be the projection of multiple 3D points, as explained earlier. This mapping is basically how the PRN facelib texture is constructed.

***Why the nostril bottom is not black like its bottom***: I think this is one disadvantage of constructing texture directly from the input image. The nostril itself is invisible in the input image; therefore, there is no way that its
color in the texture could be constructed. How could we construct one thing that does not exist?

***Solution_1***:
replace black shadown pixels with skin color in the input image (left image below)
***Solution_2***:
replace black shadown pixels with skin color in the PRN texture (right image below)
![](images/.skin_texture_pipeline_images/d95e07b8.png)

## PRN facelib training data quality.

The PRN facelib uses the face dataset [300WLP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) as its training data. Each data record in the dataset consists of
- a face image in any pose.
- a set of parameters to a statistical 3D face model, which encodes the shape of the subject face. The PRN facelib uses these parameters to reconstruct the corresponding 3D face mesh, which is then transformed to the frontal orientation to create the position map, the prediction output, as presented in [this code](https://github.com/YadiraF/face3d/blob/master/examples/8_generate_posmap_300WLP.py)

**The problem** is about the quality of these parameters. They are just results of [a fitting algorithm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) that fits the 3D face statistical model to each face image; therefore, they are not reliable ground truth to train PRN facelib.

I brought up the statistical model to compare the 3D face for each face image. It is very hard to confirm that these 3D meshes match correctly with the corresponding face in the image. From this observation, it is very hard to improve precision of PRN facelib without a good training dataset.

The Oh's suggestion to add face color segmentation as auxiliary input to the PRN facelib training might not work due to ground truth is not enough.

The author of PRN facelib suggests the following ideas to improve geometry precision: "Due to the restriction of training data, the precision of reconstructed face from this demo has little detail. You can train the network with your own detailed data or do post-processing like shape-from-shading to add details."

Below are several sample visualizations of training data. For more, please download from [the link](https://drive.google.com/open?id=14rjxSCDldbglqiXwyh0GC3vBbRG5D2DA)
![](images/.skin_texture_pipeline_images/skin_texture_pipeline-87b90f4c.png)
![](images/.skin_texture_pipeline_images/11f1c347.png)

<br/>

# Improvement ideas

## Skin synthesis

Currently, the face skin color is estimated as the mean color center of the largest cluster from the clusters of all skin pixel colors in the facial region. The estimated color is often affected by two factors
- lighting: half of the face could be dark and the other half is bright
- skin detail: skin detail like bumps could affect the center mean of the largest cluster.

Another way to extract color skin is to reuse face vertex color information from the Microsoft Deep Reconstruction model which predicts parameters to a statistical face model. Specifically, the parameters include two parts: 80 shape parameters and 80 color parameters, which encode shape and color information from low to high frequencies, as depicted in below. (the model is similar to our PCA shape model, but with color components)
![](images/.skin_texture_pipeline_images/30e51cb8.png)

The idea is that we can ignore the lighting information and the high-frequency color parameters that represent high-level face details by just keeping, for example, the first 10 color parameters. We then apply clustering on vertex colors that belongs to a pre-defined skin region. Below is the result from my experiment of keeping 10, 40, and 80 colors parameters with no lighting parameters integrated. You can see the left-most image looks more smooth than the the two other images because it is reconstructed from first 10 low-frequency parameters.
![](images/.skin_texture_pipeline_images/22eae893.png)
