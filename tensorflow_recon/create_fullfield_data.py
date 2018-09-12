import numpy as np

from simulation import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

params_cone_point = {'fname': 'data_cone_256_1nm_1um.h5',
                     'theta_st': 0,
                     'theta_end': 2 * np.pi,
                     'n_theta': 500,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': 'cone_256_filled_pp',
                     'phantom_path': 'cone_256_filled_pp/phantom',
                     'probe_type': 'point',
                     'dist_to_source_cm': 1e-4,
                     'det_psize_cm': 3e-7,
                     'theta_max': PI / 15,
                     'phi_max': PI / 15
                     }

params = params_cone_point

create_fullfield_data_numpy(energy_ev=params['energy_ev'],
                            psize_cm=params['psize_cm'],
                            free_prop_cm=params['free_prop_cm'],
                            n_theta=params['n_theta'],
                            phantom_path=params['phantom_path'],
                            save_folder=params['save_folder'],
                            fname=params['fname'],
                            batch_size=params['batch_size'],
                            probe_type=params['probe_type'],
                            wavefront_initial=None,
                            theta_st=params['theta_st'],
                            theta_end=params['theta_end']
                            )