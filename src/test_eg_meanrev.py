from datetime import datetime

from simulator import *
from agent import *
from expert import *

sim = Simulator(
        agent_cls = EG, 
        expert_cls = MeanReversion,
        start_date = datetime(2000, 1, 1), 
        end_date = datetime(2017, 8, 31))

sim.setup(
        data_dir = "../data/djia_test_small/", 
        agent_args = {
            "eta": 0.05
            }, 
        expert_args = {
            "n_obs": 10, 
            "threshold": 1.5
            })

sim.run()
sim.generate_summary()
