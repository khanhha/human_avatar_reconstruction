# Head reconstruction
- __face geometry__
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

- __face texture__
  - assume that we have a texture mapping of Victoria's head, which is different from the texutre mapping of the facelib.
    - assume that the texture domain of Victoria face is a rectangle
      - this assumption is quite hard to achieve. why?
        - how do we select the 3D face area on Victoria's head to ensure that the corresponding texture domain is a rectangle?

    - assume that the texture domain of facelib is a rectangle
      - this assumption is already satisfied. However, the rectangle texture domain includes  ear. The facelib ear is quite flat while Victoria's ear is very complex.

  - if the two above assumptions hold, we just need to scale and replace the facelib texture domain by the Victoria texture domain

  - if the two above assumption do not hold, we might to solve a texture mapping on free-domain, which could be quite expensive.

  - other suggestions for texture mapping?

- __the rest of head__
  - Question: How do we ensure that the seams between deformed Victoria face and the rest of her head is smooth?
    - There could be abrupt changes from the deformed face to the rest of head. 
