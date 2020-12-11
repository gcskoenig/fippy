from rfi.backend.causality.dags import DirectedAssyclicGraph


class StructuralEquationModel:

    def __init__(self, dag: DirectedAssyclicGraph):
        self.dag = dag

    def sample(self, n):
        pass

    def conditional_pdf(self, node):
        pass


class LinearGaussianNoiseSEM(StructuralEquationModel):

    def __init__(self, dag: DirectedAssyclicGraph, coefficients):
        self.dag = dag
        self.coefficients = coefficients

    def sample(self, n):
        pass

    def joint_cov(self):
        pass

    def joint_mean(self):
        pass

    def joint_pdf(self):
        pass


class RandomGPGaussianNoiseSEM(StructuralEquationModel):


    def __init__(self, dag: DirectedAssyclicGraph, gp_sigma: float):
        self.dag = dag
        self.gp_sigma = gp_sigma