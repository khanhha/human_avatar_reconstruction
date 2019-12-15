import bpy
import bmesh

# collect vertex in a vertex group given its name    
def collect_vertex_group(obj, name):
    target_grp_idx = -1
    for grp in obj.vertex_groups:
        if grp.name == name:
            target_grp_idx = grp.index
            break
    
    vgroups = []
    if target_grp_idx != -1:
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group  == target_grp_idx:
                    vgroups.append(v.index)
                    break
    else:
        print("collect_vertex_group: vertex group ", name, "  not found")  
        
    return vgroups

#fixing hand and feet procedure
# - load original victoria, named it "vic_mesh_origin"
# - coarsely align it the deformed victoria, whose name is "vic_mesh_only_triangle",
#   so that the hands and feet align well with the corresponding hands and feet of the deformed victoria "vic_mesh_only_triangle"
# - run script to replace hand and feet vertices
# - smooth the vertices at the hand and feet boundary
# - you can use proportional editting to make the replaced vertices fit closely

vic_org_name = "vic_mesh_origin"
vic_df_name = "vic_mesh_only_triangle"
grp_hands_name = "hands"
grp_feet_name = "feet"

vic_org_obj = bpy.data.objects[vic_org_name]
vic_df_obj = bpy.data.objects[vic_df_name]

#print(vert_idxs)

vic_org_mesh = vic_org_obj.data
vic_df_mesh = vic_df_obj.data
assert len(vic_org_mesh.vertices) == len(vic_df_mesh.vertices)

vert_idxs = collect_vertex_group(vic_org_obj, grp_hands_name)
for idx in vert_idxs:
    #print(vic_org_mesh.vertices[idx].co[:])
    vic_df_mesh.vertices[idx].co[:] =  vic_org_obj.matrix_world @ vic_org_mesh.vertices[idx].co
    
feet_vert_idxs = collect_vertex_group(vic_org_obj, grp_feet_name)
for idx in feet_vert_idxs:
    #print(vic_org_mesh.vertices[idx].co[:])
    vic_df_mesh.vertices[idx].co[:] =  vic_org_obj.matrix_world @ vic_org_mesh.vertices[idx].co
    


