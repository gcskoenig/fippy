from rfi.examples.chains import confounding2, chain2
import logging

# logging.basicConfig(level=logging.INFO)

N = 10 ** 5
confounding2.sample_and_save(n=N)
chain2.sample_and_save(n=N)
