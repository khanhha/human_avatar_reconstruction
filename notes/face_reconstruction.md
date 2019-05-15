# approach 1: deform Vicotoria to match facelib output
- To deform Victoria to match facelib output, we fist need register the two face topologies. It means that we
need to find the correspondences between vic_face vertex and facelib vertices (triangles). To find these correspondences,
an geometry alignment between the two template faces needs to be calculated, which involves steps like manuall sculpting, shrink and wrap, etc.
- However, the vic_face topology does not consist of a single connected componets. Face parts like eyelid, eyebrows, teeth and tongue are disconnected
in terms of topology with respect to the main face vertices. If we deform just the main face vertice, these parts will be left untouched.
In addition, there're no correspondences for these parts on the facelib output. If there're no correspondence (no target), how could we align them?

- so we might have to cut these parts off: eyebrow, eyelids, teeth and tongue/
    - what consequences could happen if we cut them off?
        - broken texture mapping?
            - we have to rebuild texture mapping from the modified face
    - ugly topology
        - if we're careful, the topology is still good

#approach 2: replace vic_face by facelib mesh
- the facelib mesh is a very dense mesh (43.000 vertices vs 16.000 vertices of Victoria)
- the facelib mesh topology is not good for animation and texturing
- how do we handle the seams between the facelib and vic_face?
    - it seems that we also have to handle this proble with the first approach, where we just deform a part of victoria head to match the facelib mesh


# possible problems with two approaches
- how do we deform the rest of the head, after the face is deformed?
