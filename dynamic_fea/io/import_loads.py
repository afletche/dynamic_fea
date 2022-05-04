import json
import numpy as np
import matplotlib.pyplot as plt

file_path = "./quadrotor_input_profile.json"

def import_loads(mesh, file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)

    time = data["Time"]
    input_data = data["InputSchedule"]

    # Transform Data into Desired Shape: List of force vectors for each motor

    motor_1_thrust = [np.array((0,x[0])) for x in input_data]
    motor_2_thrust = [np.array((0,x[1])) for x in input_data]

    input_loads = [motor_1_thrust, motor_2_thrust]

    dynamic_loads = []
    for i, t in enumerate(time):
        # t = time[i]
        dynamic_loads_per_time = []
        x_bounds = np.array([-0.001, 0.031])
        y_bounds = np.array([0.09, 0.11])
        mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
        mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
        mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
        mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
        mask_x = np.logical_and(mask_x1, mask_x2)
        mask_y = np.logical_and(mask_y1, mask_y2)
        mask = np.logical_and(mask_x, mask_y)

        # load_nodes = mesh.nodes[mask]
        load_node_indices = np.argwhere(mask)
        load = input_loads[0][i]
        load_per_node = load/load_node_indices.shape[0]
        for node in load_node_indices[:,0]:
            dynamic_loads_per_time.append((node, load_per_node))

        x_bounds = np.array([0.469, .501])
        y_bounds = np.array([0.09, 0.11])
        mask_x1 = mesh.nodes[:,0] >= x_bounds[0]
        mask_x2 = mesh.nodes[:,0] <= x_bounds[1]
        mask_y1 = mesh.nodes[:,1] >= y_bounds[0]
        mask_y2 = mesh.nodes[:,1] <= y_bounds[1]
        mask_x = np.logical_and(mask_x1, mask_x2)
        mask_y = np.logical_and(mask_y1, mask_y2)
        mask = np.logical_and(mask_x, mask_y)

        # load_nodes = mesh.nodes[mask]
        load_node_indices = np.argwhere(mask)
        load = input_loads[1][i]
        load_per_node = load/load_node_indices.shape[0]
        for node in load_node_indices[:,0]:
            dynamic_loads_per_time.append((node, load_per_node))

        dynamic_loads.append([t, dynamic_loads_per_time])
    
    return dynamic_loads, time