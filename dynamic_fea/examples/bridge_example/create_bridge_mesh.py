import numpy as np

from lsdo_geo.core.mesh import Mesh


def create_bridge_mesh(geo, nodes_per_length_queries):

    down_direction = np.array([0., 0., -1.])


    ''' Points to be projected'''
    bot_left1 = np.array([0., 0., 10.])
    top_left1 = np.array([0., 1., 10.])
    bot_right1 = np.array([3, 0.6, 10.])
    top_right1 = np.array([3, 1., 10.])

    bot_left2 = bot_right1
    top_right2 = np.array([3.2, 1., 10.])
    bot_right2 = np.array([3.2, 0.6, 10.])
    top_left2 = top_right1

    bot_left3 = top_left2
    top_left3 = np.array([3., 1.4, 10.])
    bot_right3 = top_right2
    top_right3 = np.array([3.2, 1.4, 10.])


    '''
    Calcualtes the lengths of the edges
    '''
    top_edge_curve1_length = np.linalg.norm(top_right1-top_left1)
    bot_edge_curve1_length = np.linalg.norm(bot_right1-bot_left1)
    avg_top_bot_length1 = (top_edge_curve1_length + bot_edge_curve1_length)/2
    left_edge_curve1_length = np.linalg.norm(top_left1-bot_left1)
    right_edge_curve1_length = np.linalg.norm(bot_right1-top_right1)
    avg_left_right_length1 = (right_edge_curve1_length + left_edge_curve1_length)/2

    left_edge_curve2_length = np.linalg.norm(top_left2-bot_left2)
    top_edge_curve3_length = np.linalg.norm(top_right3-top_left3)
    left_edge_curve3_length = np.linalg.norm(top_left3-bot_left3)


    mesh_list = []
    for i, nodes_per_length in enumerate(nodes_per_length_queries):
        '''
        Uses the nodes/[L] to calculate the number of divisions.
        '''
        NODES_PER_LENGTH = nodes_per_length
        num_pts1 = [np.rint(avg_top_bot_length1*NODES_PER_LENGTH).astype(int)]
        # num_pts2 = [int(avg_left_right_length1*NODES_PER_LENGTH)]
        num_pts2 = [np.rint(left_edge_curve2_length*NODES_PER_LENGTH).astype(int)]
        num_pts5 = [np.rint(top_edge_curve3_length*NODES_PER_LENGTH).astype(int)]
        num_pts6 = [np.rint(left_edge_curve3_length*NODES_PER_LENGTH).astype(int)]
        num_pts3 = num_pts5
        num_pts4 = num_pts2


        '''Project points'''
        bot_left1_ptset, top_right_coord = geo.project_points(bot_left1, projection_direction = down_direction)
        top_left1_ptset, top_right_coord = geo.project_points(top_left1, projection_direction = down_direction)
        bot_right1_ptset, top_right_coord = geo.project_points(bot_right1, projection_direction = down_direction)
        top_right1_ptset, top_right_coord = geo.project_points(top_right1, projection_direction = down_direction)

        bot_left2_ptset, top_right_coord = geo.project_points(bot_left2, projection_direction = down_direction)
        top_left2_ptset, top_right_coord = geo.project_points(top_left2, projection_direction = down_direction)
        bot_right2_ptset, top_right_coord = geo.project_points(bot_right2, projection_direction = down_direction)
        top_right2_ptset, top_right_coord = geo.project_points(top_right2, projection_direction = down_direction)

        bot_left3_ptset, top_right_coord = geo.project_points(bot_left3, projection_direction = down_direction)
        top_left3_ptset, top_right_coord = geo.project_points(top_left3, projection_direction = down_direction)
        bot_right3_ptset, top_right_coord = geo.project_points(bot_right3, projection_direction = down_direction)
        top_right3_ptset, top_right_coord = geo.project_points(top_right3, projection_direction = down_direction)


        '''Create chord surface using linear interpolation of lead/trail curves'''
        surface_shape1 = np.append(num_pts1,num_pts2)
        top_edge_curve1 = geo.perform_linear_interpolation(top_left1_ptset, top_right1_ptset, num_pts1)
        bot_edge_curve1 = geo.perform_linear_interpolation(bot_left1_ptset, bot_right1_ptset, num_pts1)
        left_edge_curve1 = geo.perform_linear_interpolation(bot_left1_ptset, top_left1_ptset, num_pts2)
        right_edge_curve1 = geo.perform_linear_interpolation(bot_right1_ptset, top_right1_ptset, num_pts2)
        mesh_surface1 = geo.perform_2d_transfinite_interpolation(left_edge_curve1, right_edge_curve1, bot_edge_curve1, top_edge_curve1)

        surface_shape2 = np.append(num_pts3,num_pts4)
        top_edge_curve2 = geo.perform_linear_interpolation(top_left2_ptset, top_right2_ptset, num_pts3)
        bot_edge_curve2 = geo.perform_linear_interpolation(bot_left2_ptset, bot_right2_ptset, num_pts3)
        left_edge_curve2 = geo.perform_linear_interpolation(bot_left2_ptset, top_left2_ptset, num_pts4)
        right_edge_curve2 = geo.perform_linear_interpolation(bot_right2_ptset, top_right2_ptset, num_pts4)
        mesh_surface2 = geo.perform_2d_transfinite_interpolation(left_edge_curve2, right_edge_curve2, bot_edge_curve2, top_edge_curve2)

        surface_shape3 = np.append(num_pts5,num_pts6)
        top_edge_curve3 = geo.perform_linear_interpolation(top_left3_ptset, top_right3_ptset, num_pts5)
        bot_edge_curve3 = geo.perform_linear_interpolation(bot_left3_ptset, bot_right3_ptset, num_pts5)
        left_edge_curve3 = geo.perform_linear_interpolation(bot_left3_ptset, top_left3_ptset, num_pts6)
        right_edge_curve3 = geo.perform_linear_interpolation(bot_right3_ptset, top_right3_ptset, num_pts6)
        mesh_surface3 = geo.perform_2d_transfinite_interpolation(left_edge_curve3, right_edge_curve3, bot_edge_curve3, top_edge_curve3)

        bridge_mesh = Mesh(f'bridge_mesh_{nodes_per_length}')
        mesh_list.append(bridge_mesh)
        mesh_list[i].add_pointset(mesh_surface1, name="deck")
        mesh_list[i].add_pointset(mesh_surface2, name="corner")
        mesh_list[i].add_pointset(mesh_surface3, name="barrier")

    return mesh_list