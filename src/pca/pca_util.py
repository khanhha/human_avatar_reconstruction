
def load_faces(path):

   with open(path, 'r') as file:
       lines = file.readlines()
       nvert = int(lines[0].split(' ')[0])
       nface = int(lines[0].split(' ')[1])
       face_lines = lines[nvert+1:nvert+1+nface]
       faces = []
       for line in face_lines:
           fv_strs = line.split(' ')
           assert  len(fv_strs) == 4
           fv_strs = fv_strs[:3]
           fv_idxs = [int(vstr) for vstr in fv_strs]
           faces.append(fv_idxs)

       assert len(faces) == 12894

       return faces