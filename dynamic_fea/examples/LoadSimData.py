import json
import numpy as np
import matplotlib.pyplot as plt

file_path = "./quadrotor_input_profile.json"

with open(file_path, 'r') as f:
    data = json.load(f)

time = data["Time"]
input_data = data["InputSchedule"]

# Transform Data into Desired Shape: List of force vectors for each motor

motor_1_thrust = [np.array((0,x[0])) for x in input_data]
motor_2_thrust = [np.array((0,x[1])) for x in input_data]
