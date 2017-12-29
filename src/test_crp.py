from datetime import datetime

from simulator import *
from agent import *
from expert import *

sim = Simulator(
        agent_cls = ConstantRebalancer, 
        expert_cls = Dummy,
        start_date = datetime(2000, 1, 1), 
        end_date = datetime(2017, 8, 31))

sim.setup(
        data_dir = "../data/djia_test/", 
        agent_args = {}, 
        expert_args = {})

sim.run()
sim.generate_summary()
