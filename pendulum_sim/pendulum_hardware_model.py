# library imports
import numpy as np


########### hardware const ###########
# all distances in cm & masses in g

NUM_PENDULUM_COMPONENTS = 8

########### functions to find pendulum cm and inertia tensor ###########

def find_cm(masses, positions):

    total_mass = np.sum(masses)

    return total_mass, np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass

def get_tilde_matrix(vec):

    v1, v2, v3 = vec[0], vec[1], vec[2]

    return np.array([[ 0, -v3, v2 ],
                     [ v3, 0, -v1 ],
                     [ -v2, v1, 0 ]])

def parallel_axis_theorem(total_mass, inertia_tensor_cm, inertia_offset):

    offset_tilde = get_tilde_matrix(inertia_offset)

    return inertia_tensor_cm + total_mass * offset_tilde @ np.transpose(offset_tilde)

def get_pendulum_model():

    rod_length = 30.38 / 100

    rod = { 'lenght': rod_length,
            'm': 311 / 1000,
            'cm': np.array([0, rod_length / 2, 0]) }

    box = { 'bx': 10 / 100,
            'by': 10 / 100,
            'm': 30 / 1000,
            'cm': np.array([0, 0, 0]) }

    battery = { 'm': 69 / 1000,
                'cm': np.array([0, -4 / 100 , 0]) }

    arduino_nano = { 'm': 5 / 1000,
                     'cm': np.array(np.array([4 / 100, 0, 0])) }

    xbee = { 'm': 5 / 1000,
             'cm': np.array(np.array([-4 / 100, 0, 0])) } 

    hex_nut_1 = { 'm': 2 / 1000,
                  'cm': np.array(np.array([0, 4.5/ 100, 0])) } 

    hex_nut_2 = { 'm': 2 / 1000,
                  'cm': np.array(np.array([0, 5.5 / 100, 0])) } 

    pendulum_model = { 'rod': rod,
                       'box': box,
                       'battery': battery,
                       'an': arduino_nano,
                       'xbee': xbee,
                       'hx1': hex_nut_1,
                       'hx2': hex_nut_2 }

    masses = np.zeros(shape=(NUM_PENDULUM_COMPONENTS))
    positions = np.zeros(shape=(NUM_PENDULUM_COMPONENTS, 3))

    for idx, hash_pair in enumerate(pendulum_model.items()):

        _, p_comp = hash_pair

        masses[idx] = p_comp['m']
        positions[idx] = p_comp['cm']

    total_mass, pendulum_cm = find_cm(masses, positions)

    relative_positions = positions - pendulum_cm

    inertia_tensor = np.zeros(shape=(3, 3))

    for idx in range(NUM_PENDULUM_COMPONENTS):

        tilde_matrix = get_tilde_matrix(relative_positions[idx])

        inertia_tensor += -masses[idx] * tilde_matrix @ tilde_matrix 

    pivot_offset = np.array([0, rod_length, 0]) - pendulum_cm

    inertia_tensor = parallel_axis_theorem(total_mass, inertia_tensor, pivot_offset)

    pendulum_model.update({ 'total_mass': total_mass,
                            'cm': pendulum_cm ,
                            'inertia': inertia_tensor[0][0],
                            'length': pivot_offset[1]})

    return pendulum_model


if __name__ == '__main__':

    pendulum_model = get_pendulum_model()

    print(pendulum_model['total_mass'])
    print(pendulum_model['cm'])
    print(pendulum_model['inertia'])
