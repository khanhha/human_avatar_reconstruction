
def clockwiseangle_and_distance(point, org):
    refvec = np.array([0, 1])
    # Vector between point and the origin: v = p - o
    vector = [point[0] - org[0], point[1] - org[1]]
    # Length of vector: ||v||
    lenvector = np.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -np.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]

    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = np.arctan2(diffprod, dotprod)

    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * np.pi + angle, lenvector

    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def arg_sort_points_cw(points):
    center = (np.mean(points[:, 0]), np.mean(points[:, 1]))
    compare_func = lambda pair: clockwiseangle_and_distance(pair[1], center)
    points = sorted(enumerate(points), key=compare_func)
    return [pair[0] for pair in points[::-1]]


def sort_leg_slice_vertices(slc_vert_idxs, mesh_verts):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    points_0 = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the starting point of the leg contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    start_idx = np.argmin(points_0[:, 1]) + 2
    points_0 = np.roll(points_0, axis=0, shift=-start_idx)

    # concatenate two sorted part.
    sorted_points = points_0

    # map indices
    slc_sorted_vert_idxs = []
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        assert closest_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(slc_vert_idxs[closest_idx])

    return slc_sorted_vert_idxs


def sort_torso_slice_vertices(slc_vert_idxs, mesh_verts, title=''):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    # we needs to split our point array into two part because the clockwise sort just works on convex polygon. it will fail at strong concave points at crotch slice
    # sort the upper part
    mask_0 = Y >= -0.01
    X_0 = X[mask_0]
    Y_0 = Y[mask_0]
    assert (len(X_0) > 0 and len(Y_0) > 0)
    points_0 = np.concatenate([X_0[:, np.newaxis], Y_0[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    print("\t\tpart one: ", arg_points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the first point of the contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    min_y = np.inf
    min_y_idx = 0
    for i in range(points_0.shape[0]):
        if points_0[i, 0] > 0:
            if points_0[i, 1] < min_y:
                min_y = points_0[i, 1]
                min_y_idx = i
    points_0 = np.roll(points_0, axis=0, shift=-min_y_idx)

    # sort the below part
    mask_1 = ~mask_0
    X_1 = X[mask_1]
    Y_1 = Y[mask_1]
    assert (len(X_1) > 0 and len(Y_1) > 0)
    points_1 = np.concatenate([X_1[:, np.newaxis], Y_1[:, np.newaxis]], axis=1)
    arg_points_1 = arg_sort_points_cw(points_1)
    print("\t\tpart two: ", arg_points_1)
    points_1 = np.array(points_1[arg_points_1, :])

    # concatenate two sorted part.
    sorted_points = np.concatenate([points_0, points_1], axis=0)
    # map indices
    slc_sorted_vert_idxs = []
    #print("mapping points")
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        found_idx = slc_vert_idxs[closest_idx]
        assert found_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(found_idx)

    # sorted_X =  mesh_verts[slc_sorted_vert_idxs][:,0]
    # sorted_Y =  mesh_verts[slc_sorted_vert_idxs][:,1]
    # plt.clf()
    # plt.axes().set_aspect(1)
    # plt.plot(points_0[:,0], points_0[:,1], '+r')
    # plt.plot(sorted_points[:,0], sorted_points[:,1],'-b')
    # plt.plot(sorted_X, sorted_Y,'-r')
    # plt.title(title)
    # plt.show()
    return slc_sorted_vert_idxs


def is_torso_slice(id):
    torso_slc_ids = {'Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_2', 'Hip', 'Waist', 'UnderBust',
                     'Aux_Hip_Waist_0',
                     'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1',
                     'Aux_UnderBust_Bust_0', 'Bust', 'Armscye', 'Aux_Armscye_Shoulder_0', 'Shoulder'}
    if id in torso_slc_ids:
        return True
    else:
        return False


def is_leg_slice(id):
    leg_slc_ids = {'LKnee', 'RKnee', 'LUnderCrotch', 'RUnderCrotch', 'LAux_Knee_UnderCrotch_3',
                   'LAux_Knee_UnderCrotch_2', 'LAux_Knee_UnderCrotch_1', 'LAux_Knee_UnderCrotch_0'}
    if id in leg_slc_ids:
        return True
    else:
        return False
