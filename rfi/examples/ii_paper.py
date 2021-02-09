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
        coeff_dict={'x2': {'x1': 1.0}, 'x3': {'x4': 1.0}, 'y': {'x2': 1.0, 'x3':1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_high, 'x4': sigma_high,'y': sigma_low}
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
            adjacency_matrix=np.array([[0, 1, 1, 0, 0],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0]]),
            var_names=['x1', 'x2', 'x3', 'x4', 'y']
        ),
        coeff_dict={'x2': {'x1': 1.0}, 'x3': {'x1': 1.0}, 'x4': {'x2':1.0, 'x3':1.0}, 'y': {'x4': 1.0}},
        noise_std_dict={'x1': sigma_high, 'x2': sigma_high, 'x3': sigma_high, 'x4': sigma_high,'y': sigma_low}
    )
)