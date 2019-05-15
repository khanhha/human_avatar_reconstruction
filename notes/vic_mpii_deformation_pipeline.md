# Overview
- to deform all Victoria to match all mpii-caesar meshes, first we need to calculate a parameterization of victoria mesh with respect to the mpii template mesh. Specifically, one vertex of Victoria will be represented with respect to multiple close triangles of MPII mesh.

- The calculation of parameterization requires two template meshes are well aligned. Vertices of one mesh should be exactly on the surface of the other mesh and landmark vertices should be as close as possible to the corresponding landmark vertices.

# Alignment procedure
- __Apply skeleto-based deformation to coarsely match Victoria mesh with mpii mesh__
  - turn on the "rigging:Rigify" add-in Blender
    - Files -> User Preferences -> Add-Ons -> Search for "rigging"
  - add a skeleton to Victoria
    - add -> Armature -> Basic -> Basic Human
  - scale and move skeleton in object mode to fit Victoria
  - switch to edit mode to adjust skeleton joints
    - trick: press 1 and 3 to turn to side and front view. Adjusting joint vertices in the two views are easier.
  - calculate skeleton-deformation weights
    - select the skeleton
    - hold Shift and select the Victoria mesh
    - press "Ctrl + P " => "With automatic weights"
  - select skeleton, switch to "Pose Mode"
  - rotate, translate, scale joints to deform Vic to match MPII mesh

- __Process Armpit, Crotch area__
  - the Nonrigid ICP defomration algorithm couldn't handle complex areas like Armpit and Crotch. We have to manually sculpt these areas to make the as close as possible to the corresponding areas of MPII mesh
  - select Victoria
  - switch to "Sculpt Mode"
  - use a sculpt Brushes like Inflate, Grab, Flatten to modify crotch, armpit.

- __Apply [NICP](NICP) (nonrigid iterative closest point)__
  - export MPII and Victoria to obj files
  - open NICP matlab project
  - change the path in register_mpii.m to point to mpii and victoria mesh

  - __IMPORTANT NOTE__: Victoria topology is much more dense and complex than MPII mesh. It causes the numerical optimization of the NICP algorithm fails. So instead of deforming Vic to match mpii, we do the opposite: deforming MPII to match Vic by setting Vic as the target mesh and MPII as the source mesh
  - run register_mpii.m

- __Fine-tuning using Blender modifier: Shrink and Wrap__
After NICP is applied, two mesh are still not close enough. We we shrink wrap modifier to project every vertices of MPII onto Victoria's surface
  - import MPII and Vic mesh back into Blender.
    - __IMPORTANT NOTE__: remember to set the obj import flag to "Keep Vert Order"; otherwise, Blender will chance the vertex order of the mesh, which of course change the topology data.
  - select MPII mesh
  - add modifier: Shrinkwrap
  - select "Nearest Surface Point"
  - apply the modifier


# Parameterization calculation
- run /deformation/ffdt_deformation_parameterize_tool.py

# Deformation
- run caesar/transfer_mpii_caesar_shapes.py
