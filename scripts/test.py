"""  
time_step = 1e-4

[protocol]
pre_stimulus_period = 2.0
post_stimulus_period = 0.0
stimulus_duration = 2.0
"""
import tomllib
import numpy as np
from src.coupled_columns import CoupledColumns

import matplotlib.pyplot as plt

with open('config/simulation.toml', 'rb') as f:
    simulation_parameters = tomllib.load(f)

time_step = simulation_parameters['time_step']
pre_stimulus_period = simulation_parameters['protocol']['pre_stimulus_period']
post_stimulus_period = simulation_parameters['protocol'][
    'post_stimulus_period']
stimulus_duration = simulation_parameters['protocol']['stimulus_duration']

with open('config/model.toml', 'rb') as f:
    column_parameters = tomllib.load(f)

columns = CoupledColumns(column_parameters, 'mt')

plt.imshow(columns.recurrent_weights)
plt.colorbar()
plt.show()

initial_conditions = np.zeros(2 * columns.num_populations)
feedforward_rate = np.zeros(columns.num_populations)
state = columns.simulate(feedforward_rate, initial_conditions,
                         pre_stimulus_period, time_step)

membrane_potential = state[:, :columns.num_populations]

base_line_drive = 0
delta_drive = 0

layer_4_indices = ([2, 3], [10, 11])
feedforward_rate[layer_4_indices[0]] = base_line_drive + delta_drive / 2
feedforward_rate[layer_4_indices[1]] = base_line_drive - delta_drive / 2

state = columns.simulate(feedforward_rate, state[-1], stimulus_duration,
                         time_step)

membrane_potential = np.concatenate(
    (membrane_potential, state[:, :columns.num_populations]))

plt.plot(membrane_potential)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential')
plt.show()
