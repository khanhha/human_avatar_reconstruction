After investigating make-human model, I think it's possible to replace it with Victoria head. However, I see several problems come with it.

- the face texture map of make-human causes artifacts as shown in the below picture.
Here is the reason. The nature of texture mapping is that after the 3D face is flatten into a 2D UV mapping, the distortion from 3D-to-2D triangles should be as minimal as possible. However, in the case of make-human texture map, the uv triangles at face boundary are squeezed too much. This reduces the the texture area that the corresponding 3D triangles should have and cause artifacts. In other words, the fact that a larger area of 3D triangles take up too small regions in 2D texture reduces the texture sampling quality while the model is being rendered.

Below is a set of standard face texture that I found. I think they are standard face texture maps, similar to the one in PRN facelib.

- However, I guess Jackson and the artists have their own reason to do it. Creating texture mapping this way will minimize the warping distortion from the facial part of the customer front image to the texture. I am correct?

- Another problem with Make-human head model is that if we use the whole make-human head (neck), we will lose the neck shape of the customer. Therefore, we will have to add another step of deforming the make-human neck to match customer neck.  

 I will now focus on how to warp the customer face image to a texture map (the fact that this texture map is of make-human or facelib coud be discused later). I think it is the most important task we should solve now.  
