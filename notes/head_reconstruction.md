# Head reconstruction
# Pipeline
<img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20libface_prn_network[label=%22libface%20prn%20network%22%20shape=box]%20libface_prn_output[label=%22libface%20prn%20output%22%20shape=box]%20deform_vic_face_to_match_prn_output[label=%22deform%20vic%20face%20to%20match%20prn%20output%22%20shape=box]%20head_alignment[label=%22head%20alignment%22%20shape=box]%20face_embed[label=%22face%20embedding%22%20shape=box]%20seam_solving[label=%22seam%20solving%22%20shape=box]%20head_alignment-%3Eface_embed%20face_embed%20-%3E%20seam_solving%20libface_prn_network-%3Elibface_prn_output%20libface_prn_output-%3Edeform_vic_face_to_match_prn_output%20deform_vic_face_to_match_prn_output-%3Eface_embed%20}%27'/>

## deform vic face to match prn output
use pre-calculated paramterization to deform victoria face to match prn output. We denote the output as __customer_face__

## head alignment
- use max-min along z axis to scale victoria head to match customer head (the mesh output from pca model)
- translate the scaled head to the center of the customer head
- we denote the output as __scaled_head__

## face embedding
- use facial landmarks (eye, lips) to embed (scale + rotate) the __customer_face__ face to the __scaled_head__

## seam solving
- after the __customer_face__ is is embedded to the __scaled_head__, there are discontinuities at two regions
  - neck seam between the head and the rest of body
  - face seam between the facial part and the rest of the head
- we treat the vertices at neck seam and the facial vertices as the deformation handles and solve for the remainving vertices of the head.

# Victoria face vs PRN output face alignment
## Approach 1
- __Overview__
If we follow the approach of reusing the output from the facelib, which has different topology, the head reconstruction task could be decomposed into 3 sub-tasks.
  - face geometry:  transfer the shape (geometry) from faelib topology to victoria face topology
  - face texture: transfer the texture.
  - face head seams: Process the seams between the deformed Victoria and the rest of her head

- __Face geometry__
  - __preprocessing stage__
    - coarsely translate and scale Facelib mesh to match Victoria face.
    - Mark an area on Victoria's head that defines her face area, corresponding to the area represented by the facelib mesh.
    - create a separate obj object from this triangle set
      - remember the mapping to merge it back later
    - export the vic face object to obj file
    - export the modifed facelib object to obj file
    - use Nonrigid Iterative Closest Point to deform facelib to match Vic face.
    - calculate paramterization of Victoria face with respect to facelib mesh

  - __real time stage__
    - calculate triangle coordinate basis of all triangle in the facelib output
    - deform Victoria face vertices based on
      - the parameterization
      - the calculated triangle bases

- __Face texture__
  - assume that we have a texture mapping of Victoria's head, which is different from the texutre mapping of the facelib.
    - assume that the texture domain of Victoria face is a rectangle
      - this assumption is quite hard to achieve. why?
        - how do we select the 3D face area on Victoria's head so that the corresponding texture domain is a rectangle?

    - assume that the texture domain of facelib is a rectangle
      - this assumption is already satisfied. However, the rectangle texture domain includes  ear. The facelib ear is quite flat while Victoria's ear is very complex.

  - if the two above assumptions hold, we just need to scale and replace the facelib texture domain by the Victoria texture domain

  - if the two above assumption do not hold, we might to solve a texture mapping on free-domain, which could be quite expensive.

  - other suggestions for texture mapping?

- __Face head seams__
  - Question: How do we ensure that the seams between deformed Victoria face and the rest of her head is smooth?
    - There could be abrupt changes from the deformed face to the rest of head.
    - The head, which is result from the body shape CNN, could be at a different scale or distorted further away from than the true head. In other words, the head we get from the body model might be incorrect.
      - if this case occurs, we have to deform the head.

## Approach 2
I don't know what are the other approaches yet. There is a 2019 paper that presents a full pipeline of reconstructing the whole head, but I haven't read it carefully yet.
- 2019 - Combining 3D Morphable Models: A Large scale Face-and-Head Model
