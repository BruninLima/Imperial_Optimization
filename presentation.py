import os
import random
import numpy as np
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class RawAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RawAgent, self).__init__()
        self.base_top_left = None

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type 
                and unit.alliance == features.PlayerRelative.SELF]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type 
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(RawAgent, self).step(obs)

        if obs.first():
            nexus = self.get_my_units_by_type(obs, units.Protoss.Nexus)[0]
            self.base_top_left = (nexus.x < 32)
        
        nexi   = self.get_my_units_by_type(obs, units.Protoss.Nexus)
        probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
        pylons = self.get_my_units_by_type(obs, units.Protoss.Pylon)
        completed_pylons = self.get_my_completed_units_by_type(obs,
                                                               units.Protoss.Pylon)

        free_supply = (obs.observation.player.food_cap - 
        obs.observation.player.food_used)

        if len(pylons) == 0 and obs.observation.player.minerals >= 100:
            probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                pylon_xy = (21, 22) if self.base_top_left else (42, 44)
                distances = self.get_distances(obs, probes, pylon_xy)
                probe = probes[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
           
        idle_probes = [probe for probe in probes if probe.order_length == 0]
        
        if len(idle_probes) > 0:
            probe = random.choice(idle_probes)            
            mineral_patches = [unit for unit in obs.observation.raw_units
            if unit.unit_type in [units.Neutral.BattleStationMineralField,
                                  units.Neutral.BattleStationMineralField750,
                                  units.Neutral.LabMineralField,
                                  units.Neutral.LabMineralField750,
                                  units.Neutral.MineralField,
                                  units.Neutral.MineralField750,
                                  units.Neutral.PurifierMineralField,
                                  units.Neutral.PurifierMineralField750,
                                  units.Neutral.PurifierRichMineralField,
                                  units.Neutral.PurifierRichMineralField750,
                                  units.Neutral.RichMineralField,
                                  units.Neutral.RichMineralField750]]
            
            distances = self.get_distances(obs, mineral_patches, (probe.x, probe.y))
            mineral_patch = mineral_patches[np.argmin(distances)] 
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
          "now", probe.tag, mineral_patch.tag)
            
            
        if free_supply > 1 and len(probes) < 18*len(nexi) and len(nexi) > 0:
            nexus = random.choice(nexi)
            if obs.observation.player.minerals >= 50 and nexus.order_length < 3:
                return actions.RAW_FUNCTIONS.Train_Probe_quick("now", nexus.tag)
        
        if obs.observation.player.minerals >= 100 and free_supply < 3:
            probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                x = random.randint(0,23)
                y = random.randint(0,23)
                pylon_xy = (x, y) if self.base_top_left else (64-x, 64-y)
                distances = self.get_distances(obs, probes, pylon_xy)
                probe = probes[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, np.array(pylon_xy)+np.array([probe.x//2,probe.y//2]))
        
        if obs.observation.player.minerals >= 400 and len(probes) >= 18*len(nexi):
            probes = self.get_my_units_by_type(obs, units.Protoss.Probe)
            if len(probes) > 0:
                
                x = random.randint(0,30)
                y = random.randint(0,30)
                
                if self.base_top_left:
                    nexus_xy = (x, y) 
                else:
                    nexus_xy = (64-x, 64-y)
                    
                distances = self.get_distances(obs, probes, nexus_xy)
                probe = probes[np.argmin(distances)]
                k     = random.randint(0,10)
                if k < 4:
                    return actions.RAW_FUNCTIONS.Build_Nexus_pt("now", probe.tag, np.array(nexus_xy)+np.array([probe.x,probe.y]))

        return actions.RAW_FUNCTIONS.no_op()


def main(unused_argv):
    agent = RawAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="AbyssalReef",
                players=[sc2_env.Agent(sc2_env.Race.protoss), 
                sc2_env.Bot(sc2_env.Race.protoss, 
                sc2_env.Difficulty.hard)],
                agent_interface_format=features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64,
                ),
            ) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
  app.run(main)