from fippy.backend.datagen import DirectedAcyclicGraph

def test_trivial():
    dag = DirectedAcyclicGraph.random_dag(10, 0.3)
    assert True