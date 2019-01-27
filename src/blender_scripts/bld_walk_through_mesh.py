bl_info = {
    "name": "Add-on Template",
    "description": "",
    "author": "",
    "version": (0, 0, 1),
    "blender": (2, 70, 0),
    "location": "3D View > Tools",
    "warning": "", # used for warning icon and text in addons panel
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"
}

from pathlib import Path
import numpy as np
from mathutils import Vector
from pathlib import Path
import pickle as pkl
import math
import os
from collections import defaultdict
import shutil

import bpy

from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
                       
from bpy.types import (Panel,
                       Operator,
                       PropertyGroup,
                       )


class MySettings(PropertyGroup):
    
    current_idx = IntProperty(
        name="current_idx",
        description=":",
        default=-1,
        min = -1,
        max = 1000000
    )
    
    current_name = StringProperty(
        name = "current_name",
        description = "current caesar obj name",
        default = ''
    )
    
    request_name = StringProperty(
        name = "request_name",
        description = "some object name specified by users",
        default = ''
    )
    
    dir_caesar_mesh = StringProperty(
        name = "dir_caesar_mesh",
        description = "directory contain caesar mesh"
    )
    
    dir_ctl_mesh = StringProperty(
        name = "dir_ctl_mesh",
        description = "directory contain control mesh"
    )
    
    dir_df_mesh = StringProperty(
        name = "dir_df_mesh",
        description = "directory contain deformed mesh"
    )
        
    caesar_names = []
    ctl_names = []
    df_names = []
    ld_idxs = []

def import_obj(path, name):
    bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z', split_mode='OFF')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj

def delete_obj(obj):
    select_single_obj(obj)
    bpy.ops.object.delete()
    
def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    
def transform_obj_caesar(obj, ld_idxs):
    mesh = obj.data
    
    s = 0.01
    bpy.ops.transform.resize(value=(s,s,s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    
    org = mesh.vertices[ld_idxs[72]].co
    all_z = [v.co[2] for v in mesh.vertices]
    min_z = np.min(np.array(all_z))
    org[2] = min_z
    
    bpy.ops.transform.translate(value = -org)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
       
    p0_x = mesh.vertices[ld_idxs[16]].co
    p1_x = mesh.vertices[ld_idxs[18]].co    
    x = p1_x - p0_x
    x.normalize()
    angle = x.dot(Vector((1.0, 0.0, 0.0)))
    #angle =  math.degrees(math.acos(angle))
    angle = math.acos(angle)
    print('angle', angle)
    bpy.ops.transform.rotate(value=angle, axis=(0.0,0.0,1.0))
    
    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

def load_obj_names(dir):
    names = []
    for path in Path(dir).glob("*.obj"):
        names.append(path.stem)
    return sorted(names)

def delete_current_objects():
    for obj in bpy.data.objects:
        if "caesar_mesh_" in obj.name:
            delete_obj(obj)
            
        if "ctl_mesh_" in obj.name:
            delete_obj(obj)

        if "df_mesh_" in obj.name:
            delete_obj(obj)
            
def load_objects(name_id, my_tool):
    t = my_tool
    
    delete_current_objects()
    
    for name in t.caesar_names:
        if name_id in name:
            caesar_path = t.dir_caesar_mesh + '/' + name + '.obj'
            if Path(caesar_path).exists():
                caeobj = import_obj(caesar_path, 'caesar_mesh_'+name)
                transform_obj_caesar(caeobj, t.ld_idxs[0])
                print('loaded caesar mesh: ', name)
            else:
                print('path ' + caesar_path + ' does not exist');
    
    ok = False
    for name in t.ctl_names:
        if name_id in name:
            ctl_path = t.dir_ctl_mesh + '/' + name + '.obj'
            if Path(ctl_path).exists():
                ctlobj = import_obj(ctl_path, 'ctl_mesh_')
                select_single_obj(ctlobj)
                
                s = 1.1
                bpy.ops.transform.resize(value=(s,s,s))
                bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    
                bpy.ops.transform.translate(value = Vector((8.0, 0.0, 0.0)))
                bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
                
                print('loaded ctl mesh: ', name)
                ok = True    
            break
    if ok == False:
        print('cannot find ctl mesh with prefix name: ', name_id)
        
    ok = False
    for name in t.df_names:
        if name_id in name:
            df_path = t.dir_df_mesh + '/' + name + '.obj'              
            if Path(df_path).exists(): 
                dfobj = import_obj(df_path, 'df_mesh_')
                print('loaded df mesh: ', name)
                ok = True
            break
    if ok == False:
        print('cannot find deform mesh with prefix name: ', name_id)
    
class InitData(bpy.types.Operator):
    bl_idname = "wm.init_data"
    bl_label = "initialize"
    
    def init_data(self, mytool):
        print('init_data')
        ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
        ld_idxs = np.load(ld_path)
        mytool.ld_idxs.clear()
        mytool.ld_idxs.append(ld_idxs)
        
        names = load_obj_names(mytool.dir_caesar_mesh)
        mytool.caesar_names.clear()
        for name in names:
            mytool.caesar_names.append(name)
            
        names = load_obj_names(mytool.dir_ctl_mesh)
        mytool.ctl_names.clear()
        for name in names:
            mytool.ctl_names.append(name)
        
        names = load_obj_names(mytool.dir_df_mesh)
        mytool.df_names.clear()
        for name in names:
            mytool.df_names.append(name)
            
        print('n caesar mesh:', len(mytool.caesar_names))
        print('n ctl mesh:', len(mytool.ctl_names))
        print('n df mesh:', len(mytool.df_names))
        
        mytool.current_idx = 0
        if len(mytool.caesar_names) > 0:
            mytool.current_name = mytool.caesar_names[mytool.current_idx]
        
        load_objects(mytool.current_name, mytool)
        
        
    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool
        self.init_data(mytool)
        
        return {'FINISHED'}  
        
class NextObjectOperator(bpy.types.Operator):
    bl_idname = "wm.next_object"
    bl_label = "next object"

    def execute(self, context):
        scene = context.scene
        t = scene.my_tool
              
        t.current_idx += 1
        print(t.ld_idxs)
        if len(t.caesar_names) == 0:
            print("empty caesar file names")
            return {'FINISHED'}
        
        cur_name = t.caesar_names[t.current_idx]
        t.current_name = cur_name
        load_objects(cur_name, t)        
        return {'FINISHED'}

class LoadObjectByNameOperator(bpy.types.Operator):
    bl_idname = "wm.load_object_by_name"
    bl_label = "load object by name"

    def execute(self, context):
        scene = context.scene
        t = scene.my_tool
        if t.request_name != '':
            print('start loading object ', t.request_name)
            load_objects(t.request_name, t)        
        return {'FINISHED'}
    
# ------------------------------------------------------------------------
#    my tool in objectmode
# ------------------------------------------------------------------------
class OBJECT_PT_MY_PANNEL(Panel):
    bl_idname = "OBJECT_PT_MY_PANNEL"
    bl_label = "object walk through"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "scene"

    @classmethod
    def poll(self,context):
        #return context.object is not None
        return True
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool

        layout.prop(mytool, "dir_caesar_mesh")
        layout.prop(mytool, "dir_ctl_mesh")
        layout.prop(mytool, "dir_df_mesh")
        layout.operator("wm.init_data")
        layout.prop(mytool, "current_idx")
        layout.prop(mytool, "current_name")
        layout.operator("wm.next_object")
        layout.prop(mytool, "request_name")
        layout.operator("wm.load_object_by_name")

# ------------------------------------------------------------------------
# register and unregister
# ------------------------------------------------------------------------

def register():
    bpy.utils.register_module(__name__)
    bpy.types.Scene.my_tool = PointerProperty(type=MySettings)
    my_tool = bpy.context.scene.my_tool
    my_tool.current_idx = -1
    my_tool.current_name = ''

def unregister():
    bpy.utils.unregister_modsule(__name__)
    del bpy.types.Scene.my_tool

if __name__ == "__main__":
    register()