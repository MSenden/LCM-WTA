import numpy as np
import matplotlib.pyplot as plt

from src.coupled_columns import CoupledColumns
from src.utils import load_config, create_feedforward_input, compute_firing_rate

if __name__ == '__main__':

    # Load configurations
    simulation_parameters = load_config('config/simulation.toml')
    column_parameters = load_config('config/model.toml')

    # Extract parameters
    time_step = simulation_parameters['time_step']
    protocol = simulation_parameters['protocol']

    initial_conditions = np.concatenate(
        (simulation_parameters['initial_conditions']['membrane_potential'],
         simulation_parameters['initial_conditions']['adaptation']))

    layer_4_indices = column_parameters['layer_4_indices']

    # Initialize columns
    columns = CoupledColumns(column_parameters, 'mt')
    states_list = []

    initial_membrane_potential = simulation_parameters['initial_conditions'][
        'membrane_potential']
    initial_adaptation = simulation_parameters['initial_conditions'][
        'adaptation']
    initial_conditions = np.concatenate(
        (initial_membrane_potential, initial_adaptation))

    # Pre-stimulus phase
    feedforward_rate = np.zeros(columns.num_populations)
    state = columns.simulate(feedforward_rate, initial_conditions,
                             protocol['pre_stimulus_period'], time_step)

    # Store results
    states_list.append(state)

    # Stimulus phase
    feedforward_rate = create_feedforward_input(
        columns.num_populations, layer_4_indices,
        protocol['mean_stimulus_drive'], protocol['difference_stimulus_drive'])
    state = columns.simulate(feedforward_rate, state[-1],
                             protocol['stimulus_duration'], time_step)

    # Store results (concatenate)
    states_list.append(state)

    # Post-stimulus phase
    feedforward_rate = np.zeros(columns.num_populations)
    state = columns.simulate(feedforward_rate, state[-1],
                             protocol['post_stimulus_period'], time_step)

    # Store results (concatenate)
    states_list.append(state)

    # Convert list to numpy array
    state = np.concatenate(states_list)

    # Compute firing rate
    firing_rate = compute_firing_rate(state[:, :columns.num_populations],
                                      state[:, columns.num_populations:],
                                      columns.gain_function_parameters)
    # Plot results
    plt.figure()
    plt.plot(firing_rate)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing Rate')
    plt.show()
