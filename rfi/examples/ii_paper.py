import numpy as np

from rfi.backend.causality.dags import DirectedAcyclicGraph
from rfi.backend.causality.sem import LinearGaussianNoiseSEM
from rfi.examples import SyntheticExample

sigma_low = 0.3
sigma_medium = .5
sigma_high = 1
sigma_veryhigh = 1.5


"""
EXAMPLE 3: Figure 1 from II paper

Order: X_1, X_2, Y
"""
ii_audit = SyntheticExample(
    name='ii-model-audit',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 1],
                                       [0, 0, 0],
                                       [0, 0, 0]]),
            var_names=['x1', 'x2', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'y': {'x1': 1.0}},
        noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium, 'y': sigma_low}
    )
)

"""
EXAMPLE 4: Figure 2 from II paper

Order: X_1, X_2, Y
"""
ii_inference = SyntheticExample(
    name='ii-inference',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 0],
                                       [0, 0, 1],
                                       [0, 0, 0]]),
            var_names=['x1', 'x2', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'y': {'x2': 1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high, 'y': sigma_low}
    )
)

"""
EXAMPLE 4: Figure 2 from II paper

Order: X_1, X_2, Y
"""
ii_inference_large = SyntheticExample(
    name='ii-inference-large',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 0, 0, 0]]),
            var_names=['x1', 'x2', 'x3', 'x4', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'x3': {'x4': 1.0},
                    'y': {'x2': 1.0, 'x3': 1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high,
                        'x3': sigma_high, 'x4': sigma_high,
                        'y': sigma_low}
    )
)


"""
EXAMPLE 4: Figure 2 from II paper

Order: X_1, X_2, Y
"""
ii_rhombus = SyntheticExample(
    name='ii-inference-large',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 1, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 0, 1],
                                       [0, 0, 0, 0]]),
            var_names=['x1', 'x2', 'x3', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'x3': {'x1': 1.0},
                    'y': {'x2': 1.0, 'x3': 1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high,
                        'x3': sigma_high, 'y': sigma_medium}
    )
)


"""
EXAMPLE 5: Chaos

Order: par_par12, par1, par2, par3, y, eff_par2,
eff_par3/par_eff3, eff1, eff2, eff3, par_eff1, eff_eff12
"""
ii_chaos = SyntheticExample(
    name='ii-chaos',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            var_names=['par_par12', 'par1', 'par2', 'par3', 'y', 'eff_par2',
                       'eff_par3/par_eff3', 'eff1', 'eff2', 'eff3',
                       'par_eff1', 'eff_eff12']
        ),
        coeff_dict={'par1': {'par_par12': 1.0}, 'par2': {'par_par12': 1.0},
                    'y': {'par1': 1.0, 'par2': 1.0, 'par3': 1.0},
                    'eff_par2': {'par2': 1.0},
                    'eff_par3/par_eff3': {'par3': 1.0},
                    'eff1': {'y': 1.0},
                    'eff2': {'y': 1.0},
                    'eff3': {'y': 1.0, 'eff_par3/par_eff3': 1.0},
                    'eff_eff12': {'eff1': 1.0, 'eff2': 1.0}},
        noise_std_dict={'par_par12': sigma_high, 'par1': sigma_high,
                        'par2': sigma_high, 'par3': sigma_high,
                        'y': sigma_low, 'eff_par2': sigma_high,
                        'eff_par3/par_eff3': sigma_high, 'eff1': sigma_high,
                        'eff2': sigma_high, 'eff3': sigma_high,
                        'par_eff1': sigma_high, 'eff_eff12': sigma_high}
    )
)

"""
EXAMPLE 6: Cause and Effect

Order: X_1, X_2, Y, X_3, X_4
"""
ii_cause_effect = SyntheticExample(
    name='ii-cause_and_effect',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 1, 0, 0]]),
            var_names=['x1', 'x2', 'x3', 'x4', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'x3': {'y': 1.0},
                    'y': {'x2': 1.0}, 'x4': {'x3': 1.0}},
        noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium,
                        'x3': sigma_medium, 'x4': sigma_medium,
                        'y': sigma_high}
    )
)


"""
EXAMPLE 7: relevance_types

Order: X_1, X_2, X_3, X_4, Y
"""
ii_relevance_types = SyntheticExample(
    name='ii-relevance_types',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0, 0, 0, 1],
                                       [0, 0, 1, 1],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]]),
            var_names=['x1', 'x2', 'x3', 'y']
        ),
        coeff_dict={'x3': {'x2': 1.0},
                    'y': {'x1': 1.0, 'x2': 1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high,
                        'x3': sigma_high,
                        'y': sigma_high}
    )
)

"""
EXAMPLE 8: large
"""

ii_large = SyntheticExample(
    name='ii-large',
    sem=LinearGaussianNoiseSEM(
        dag=DirectedAcyclicGraph(
            adjacency_matrix=np.array([[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                       [0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0]]),
            var_names=['x1', 'x2', 'x3','x4','x5','x6','x7','x8','x9','x10',
                       'x11','x12','x13','x14','x15','x16','x17','x18', 'y']
        ),
        coeff_dict={'x6': {'x1': 1.0, 'x2': 1.0},
                    'x7': {'x2': 1.0, 'x3': 1.0},
                    'x8': {'x3': 1.0, 'x4': 1.0},
                    'x9': {'x4': 1.0, 'x5': 1.0},
                    'y': {'x6': 1.0, 'x7': 1.0, 'x8': 1.0, 'x9': 1.0},
                    'x10': {'y': 1.0},
                    'x11': {'y': 1.0},
                    'x12': {'y': 1.0},
                    'x13': {'y': 1.0},
                    'x14': {'x10': 1.0},
                    'x15': {'x11': 1.0, 'x10': 1.0},
                    'x16': {'x12': 1.0, 'x11': 1.0},
                    'x17': {'x13': 1.0, 'x12': 1.0},
                    'x18': {'x13': 1.0}},
        noise_std_dict={'x1': sigma_medium, 'x2': sigma_medium,
                        'x3': sigma_medium, 'x4': sigma_medium,
                        'x5': sigma_medium, 'x6': sigma_medium,
                        'x7': sigma_medium, 'x8': sigma_medium,
                        'x9': sigma_medium, 'x10': sigma_medium,
                        'x11': sigma_medium, 'x12': sigma_medium,
                        'x13': sigma_medium, 'x14': sigma_medium,
                        'x15': sigma_medium, 'x16': sigma_medium,
                        'x17': sigma_medium, 'x18': sigma_medium,
                        'y': sigma_high}
    )
)
