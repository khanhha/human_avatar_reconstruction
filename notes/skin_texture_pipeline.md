
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Pipeline](#pipeline)
  - [Run PRN facelib.](#run-prn-facelib)
  - [Adjust input image (optional stage)](#adjust-input-image-optional-stage)
    - [Idea](#idea)
    - [Problems with seamless cloning.](#problems-with-seamless-cloning)
  - [Texuture mapping](#texuture-mapping)

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

### Idea
- Apply grabcut to detect skin region. The foreground mask for grabcut is created by exclude the eye and lip polygons from the convex polygon of the 68 landmarks, and then erode with a kernel of 10x10 to make sure that there is no wall/background color inside the foreground mask.

![](images/.skin_texture_pipeline_images/grabcut.png)

- Create a background image filled with the estimated skin color.
- Apply seamless cloning to transfer the facial area with grab-cut mask to the skin background image. The middle figure shows result without cloning, just replace every pixels outside the skin mask by estimated color. The right-most figure shows the result with seamless cloning.

![](images/.skin_texture_pipeline_images/seamless_cloning.png)

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
