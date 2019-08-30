import bpy
from mathutils import Vector

bpy.ops.object.mode_set(mode = 'EDIT')
obj = bpy.data.objects['metarig']
obj.select = True
bpy.context.scene.objects.active = obj

b = obj.data.edit_bones['forearm.L']
t = Vector((4.42245, -0.522022, 8.78979))

m = obj.matrix_world * b.matrix
b.tail = obj.matrix_world.inverted() * t