# library imports
import numpy as np


########### hardware const ###########
# all distances in cm & masses in g

NUM_PENDULUM_COMPONENTS = 8

rod = { 'lenght': 5,
        'm': 2 / 1000,
        'cm': np.array([0, 3, 0]) }

box = { 'bx': 1,
        'by': 2,
        'm': 2 / 1000,
        'cm': np.array([0, 0, 0]) }

battery = { 'm': 5 / 1000,
            'cm': np.array([0, -0.3, 0]) }

arduino_nano = { 'm': 1 / 1000,
                 'cm': np.array(np.array([0.4, 0, 0])) }

xbee = { 'm': 1 / 1000,
         'cm': np.array(np.array([-0.4, 0, 0])) } 

hex_nut_1 = { 'm': 1.5 / 1000,
              'cm': np.array(np.array([0, 0.45, 0])) } 

hex_nut_2 = { 'm': 1.5 / 1000,
              'cm': np.array(np.array([0, 0.55, 0])) } 

pendulum_model_const = { 'rod': rod,
                         'box': box,
                         'battery': battery,
                         'an': arduino_nano,
                         'xbee': xbee,
                         'hx1': hex_nut_1,
                         'hx2': hex_nut_2 }

########### functions to find pendulum cm and inertia tensor ###########

def find_cm(masses, positions):

    total_mass = np.sum(masses)

    return total_mass, np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass

def get_tilde_matrix(vec):

    v1, v2, v3 = vec[0], vec[1], vec[2]

    return np.array([[ 0, -v3, v2 ],
                     [ v3, 0, -v1 ],
                     [ -v2, v1, 0 ]])

def find_inertia_tensor(pendulum_model_const):

    masses = np.zeros(shape=(NUM_PENDULUM_COMPONENTS))
    positions = np.zeros(shape=(NUM_PENDULUM_COMPONENTS, 3))

    for idx, hash_pair in enumerate(pendulum_model_const.items()):

        _, p_comp = hash_pair

        masses[idx] = p_comp['m']
        positions[idx] = p_comp['cm']

    total_mass, pendulum_cm = find_cm(masses, positions)

    relative_positions = positions - pendulum_cm

    inertia_tensor = np.zeros(shape=(3, 3))

    for idx in range(NUM_PENDULUM_COMPONENTS):

        tilde_matrix = get_tilde_matrix(relative_positions[idx])

        inertia_tensor += -masses[idx] * tilde_matrix @ tilde_matrix 

    pendulum_model = { 'total_mass': total_mass,
                       'cm': pendulum_cm ,
                       'inertia': inertia_tensor}

    return pendulum_model


if __name__ == '__main__':

    pendulum_model = find_inertia_tensor(pendulum_model_const)

    print(pendulum_model['total_mass'])
    print(pendulum_model['cm'])
    print(pendulum_model['inertia'])
