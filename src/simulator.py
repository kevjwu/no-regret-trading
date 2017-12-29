import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", family = "serif", serif = "cmr10") 

#
# Simulator
# 


class Simulator(object):

    def __init__(
            self,
            agent_cls, expert_cls, 
            start_date, end_date, 
            wealth=1.0, 
            price_col="adj_close"):

        self.agent_cls = agent_cls
        self.expert_cls = expert_cls
        self.start_date = start_date
        self.end_date = end_date
        self.wealth = wealth
        self.init_wealth = wealth
        self.current_date = None
        self.price_col = price_col

        self.weight_history = []
        self.reward_history = {}
        self.buy_history = {}
        self.price_history = {}
        self.wealth_history = []
        self.dates = []

        self.summary = dict(
                    initial_wealth = self.wealth,
                    agent_type = str(self.agent_cls),
                    expert_type = str(self.expert_cls)
                )

        return None

    def setup(
            self, data_dir, 
            agent_args = {},
            expert_args = {}):

        self.data = {f.replace(".csv", "").upper(): pd.read_csv(data_dir+f, iterator=True, chunksize=1) 
                for f in os.listdir(data_dir)}
        self.assets = self.data.keys()
        self.experts = {asset: self.expert_cls(asset, **expert_args) 
                for asset in self.assets}

        self.buy_history = {asset: [] for asset in self.assets}
        self.price_history = {asset: [] for asset in self.assets}
        self.reward_history = {asset: [] for asset in self.assets}

        self.agent = self.agent_cls(self.experts, **agent_args)
        self.portfolio = np.array([weight * self.wealth 
            for weight in self.agent.weights])

        self.summary["agent_args"] = agent_args
        self.summary["expert_args"] = expert_args

        return None

    def run(self):
        while True:
            try:
                for asset in self.assets:
                    d = self.data[asset].get_chunk(1)
                    self.experts[asset].update(dict(d)[self.price_col])

                self.current_date = datetime.strptime(d["date"].item(), "%Y-%m-%d")
                if self.current_date > self.end_date:
                    break

                self.log()

                ## Agent updates weights
                self.agent.update()

                ## Rewards are accrued to the portfolio
                if not np.all(self.agent.rewards):
                    continue

                self.portfolio = self.portfolio * self.agent.rewards
                self.wealth = np.sum(self.portfolio)

                self.portfolio = np.array([weight * self.wealth for weight in self.agent.weights])

                ## TODO: If expert doesn't pick, reallocate portfolio
            except StopIteration:
                break



    def log(self):
        self.weight_history.append(self.agent.weights)
        self.dates.append(self.current_date)
        for asset in self.assets:
            self.buy_history[asset].append(self.experts[asset].buy)
            self.reward_history[asset].append(self.experts[asset].reward)
            self.price_history[asset].append(self.experts[asset].last_price)
        self.wealth_history.append(self.wealth)

        return None

    def generate_summary(self):
  
        subdir = datetime.strftime(datetime.now(), "%Y-%m-%d_%H%M%S")
        os.makedirs(os.path.join("results", subdir))

        labels = self.assets

        self.summary["final_wealth"] = self.wealth
        self.summary["start_date"] = datetime.strftime(self.dates[0], "%Y-%m-%d")
        self.summary["end_date"] = datetime.strftime(self.dates[-1], "%Y-%m-%d")
        self.summary["annualized_return"] = (self.wealth/self.init_wealth)**(252./len(self.dates)) - 1.

        with open(os.path.join("results", subdir, "summary.json"), "w") as f:
            json.dump(self.summary, f)

        ## Wealth summmary 

        df = pd.Series(self.wealth_history)
        plt.plot(df)
        plt.ylabel("Wealth")
        plt.xlabel("Round")
        plt.ticklabel_format(useOffset=False)
        plt.savefig(os.path.join("results", subdir, "wealth.png"), bbox_inches="tight", dpi=300)
        plt.close()

        df.index = self.dates
        df.to_csv(os.path.join("results", subdir, "wealth.csv"))


        ## Rewards summary

        df = pd.DataFrame(self.reward_history)
        df = df.replace(0, 1.0)
        plt.plot(df)
        plt.ylabel("Rewards")
        plt.xlabel("Round")
        plt.ticklabel_format(useOffset=False)
        plt.legend(labels, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.savefig(os.path.join("results", subdir, "rewards.png"), bbox_inches="tight", dpi=300)
        plt.close()
        
        df.index = self.dates
        df.to_csv(os.path.join("results", subdir, "rewards.csv"))

        ## Weights summary

        df = pd.DataFrame(self.weight_history)
        plt.plot(df)
        plt.ylabel("Weights")
        plt.xlabel("Round")
        plt.ticklabel_format(useOffset=False)
        plt.legend(labels, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.savefig(os.path.join("results", subdir, "weights.png"), bbox_inches="tight", dpi=300)
        plt.close()

        df.index = self.dates
        df.to_csv(os.path.join("results", subdir, "weights.csv"))


        ## Returns summary

        df = pd.DataFrame(self.price_history)
        df = df.divide(df.ix[0])
        plt.plot(df)
        plt.ylabel("Returns")
        plt.xlabel("Round")
        plt.ticklabel_format(useOffset=False)
        plt.legend(labels, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.savefig(os.path.join("results", subdir, "returns.png"), bbox_inches="tight", dpi=300)
        plt.close()

        df.index = self.dates
        df.to_csv(os.path.join("results", subdir, "returns.csv"))

        return None







