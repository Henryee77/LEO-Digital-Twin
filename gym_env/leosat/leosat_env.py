"""leosat_env.py"""
from __future__ import annotations
from typing import List, Set, Dict, Tuple, Any
import random
import csv
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
from low_earth_orbit.ground_user import User
from low_earth_orbit.constellation import Constellation
from low_earth_orbit.constellation import ConstellationData
from low_earth_orbit.nmc import NMC
from low_earth_orbit.util import util
from low_earth_orbit.util import Position
from low_earth_orbit.util import Geodetic
from low_earth_orbit.util import constant
from MA_TD3.agent.agent import Agent


class LEOSatEnv(gym.Env):
  """The LEO Env class."""

  ue_dict: Dict[str, User]

  def __init__(self,
               ax: plt.Axes,
               args,
               tb_writer: SummaryWriter,
               agent_dict: Dict[str, Agent],
               real_agents: Dict[str, Agent],
               digital_agents: Dict[str, Agent],
               agent_names: List[str]):
    super(LEOSatEnv, self).__init__()
    self.name = 'LEOSat'
    self.ax = ax
    self.args = args
    self.tb_writer = tb_writer
    self.constel = self.ues = self.nmc = None
    self.ue_dict = {}
    self.prev_cell_sinr = {}
    self.prev_beam_power = {}
    self.ee = {}
    self.data_rate = {}  # Consider overhead
    self.throughput = {}  # No overhead
    self.ue_throughput = {}
    self.penalty = {}
    self.overhead = {}
    self.ue_pos_data = {}
    self.load_ues_data()

    self.online = False
    self.twin_online = False
    self.leo_agents = agent_dict
    self.real_agents = real_agents
    self.digital_agents = digital_agents
    self.agent_num = len(agent_names)
    self.agent_names = agent_names
    self.cell_num = 0

    self.step_num = 0
    self.total_step_num = 0
    self.action_period = args.action_timeslot / constant.MOVING_TIMESLOT
    self.reset_count = 0
    self.max_step = args.max_time_per_ep
    self.last_episode = False
    self.plot_range = 2.5  # the plotting range

    self.action_space = spaces.Box(np.array([-1]), np.array([1]))  # dummy for gym template
    self.observation_space = spaces.Box(np.array([-10]), np.array([10]))

    self.max_power = constant.MAX_POWER

    self.additional_beam_set = set()
    self.random_interfer_beam_num = 2
    self.interference_change_freq = 5
    self.interfer_power_low = 20
    self.interfer_power_high = util.truncate(
      self.max_power - util.todb(self.random_interfer_beam_num))
    self.overflowed_overhead = {}
    for sat_name in self.agent_names:
      self.overflowed_overhead[sat_name] = 0

    if self.leo_agents is not None:
      self._init_env()

  @ property
  def name(self):
    return self._name

  @ name.setter
  def name(self, name):
    self._name = name

  @ property
  def leo_agents(self) -> Dict[str, Agent]:
    return self.__leo_agents

  @ leo_agents.setter
  def leo_agents(self, agent_dict: Dict[str, Agent]):
    self.__leo_agents = agent_dict

  @ property
  def real_agents(self) -> Dict[str, Agent]:
    return self._real_agents

  @ real_agents.setter
  def real_agents(self, agent_dict: Dict[str, Agent]):
    if agent_dict is None:
      raise ValueError('Digital agents cannot be None type')
    self._real_agents = agent_dict

  @ property
  def digital_agents(self) -> Dict[str, Agent]:
    return self._digital_agents

  @ digital_agents.setter
  def digital_agents(self, agent_dict: Dict[str, Agent]):
    if agent_dict is None:
      raise ValueError('Digital agents cannot be None type')
    self._digital_agents = agent_dict

  def set_online(self, self_online, twin_online):
    self.online = self_online
    self.twin_online = twin_online

  def step(self, action_n: Dict[str, npt.NDArray]) -> Tuple[Dict[str, npt.NDArray[np.float32]], Dict[str, float], bool, bool, Any]:
    self._take_action(action_n)

    ue_sinr = self.constel.cal_transmission_sinr(ues=self.ues,
                                                 interference_beams=self.additional_beam_set)
    ue_throughput = self.constel.cal_throughput(ues=self.ues,
                                                sinr=ue_sinr,
                                                interference_beams=self.additional_beam_set)
    for ue_name, thput in ue_throughput.items():
      self.ue_throughput[self.step_num][ue_name] = thput

    reward = self._cal_reward(ue_throughput=ue_throughput)
    self.record_steps_of_last_ep(ue_sinr=ue_sinr, ue_throughput=ue_throughput)

    self.step_num += 1
    self.total_step_num += 1
    self.constel.update_sat_position()
    self.dynamic_channel_update()
    done = (self.step_num >= self.max_step)
    truncated = (self.step_num >= self.max_step)
    has_action = (self.step_num % self.action_period == 0)
    if has_action:
      for sat_name in self.agent_names:
        self.overflowed_overhead[sat_name] = 0
    obs = self.get_state_info(has_action)

    return (obs, reward, done, truncated, {'has_action': has_action})

  def dynamic_channel_update(self):
    pass

  def record_steps_of_last_ep(self, ue_sinr, ue_throughput):
    if self.last_episode:
      for ue_name, sinr in ue_sinr.items():
        self.tb_writer.add_scalars(f'{self.name} Env Param/ue sinr',
                                   {ue_name: sinr},
                                   self.total_step_num)
      for ue_name, throughput in ue_throughput.items():
        self.tb_writer.add_scalars(f'{self.name} Env Param/ue throughput',
                                   {ue_name: throughput},
                                   self.total_step_num)

  def save_episode_result(self):
    self.tb_writer.add_scalars(f'{self.name} Env Param/average overhead',
                               {f'{self.args.prefix} {self.name}': util.avg_time_sat_dict(self.overhead)},
                               self.reset_count)
    self.tb_writer.add_scalars(f'{self.name} Env Param/average data rate',
                               {f'{self.args.prefix} {self.name}': util.avg_time_sat_dict(self.data_rate)},
                               self.reset_count)
    self.tb_writer.add_scalars(f'{self.name} Env Param/average throughput',
                               {f'{self.args.prefix} {self.name}': util.avg_time_sat_dict(self.throughput)},
                               self.reset_count)
    self.tb_writer.add_scalars(f'{self.name} Env Param/average EE',
                               {f'{self.args.prefix} {self.name}': util.avg_time_sat_dict(self.ee)},
                               self.reset_count)
    self.tb_writer.add_scalars(f'{self.name} Env Param/average overflowed overhead',
                               {f'{self.args.prefix} {self.name}': sum(self.overflowed_overhead.values()) /
                                len(self.overflowed_overhead)},
                               self.reset_count)
    pen_saved_dict = {}
    ue_thput_saved_dict = {}
    for ue_name in self.ue_dict:
      pen_saved_dict[f'{self.args.prefix} {self.name} {ue_name}'] = sum(
        self.penalty[t][ue_name] for t in range(self.step_num)) / self.step_num
      ue_thput_saved_dict[f'{self.args.prefix} {self.name} {ue_name}'] = sum(
        self.ue_throughput[t][ue_name] for t in range(self.step_num)) / self.step_num
    self.tb_writer.add_scalars(f'{self.name} Env Param/penalty term',
                               pen_saved_dict,
                               self.reset_count)
    self.tb_writer.add_scalars(f'{self.name} Env Param/UE average throughput',
                               ue_thput_saved_dict,
                               self.reset_count)

  def _take_action(self, action_n: Dict[str, List[float]]):
    satbeam_list = []
    for sat_name, action in action_n.items():
      agent = self.leo_agents[sat_name]
      turned_on_beams = self.action_to_beam_list(action=action[agent.beam_slice])
      satbeam_list = satbeam_list + [(sat_name, beam_idx) for beam_idx in turned_on_beams]
    for ue in self.ues:
      # print(ue.servable, satbeam_list)
      ue.filter_servable(satbeam_list)
      # print(self.name, self.step_num, ue.servable)

    # beam decision and handover
    self.nmc.a3_event_check()

    dbm_power_dict = {}
    for sat_name, action in action_n.items():
      agent = self.leo_agents[sat_name]
      dbm_power_dict[sat_name] = self.action_to_power_dict(action[agent.power_slice], sat_name)

      beamwidth_dict = self.action_to_beamwidth_dict(action[agent.beamwidth_slice], sat_name)
      for beam_idx, beamwidth in beamwidth_dict.items():
        self.leo_agents[sat_name].sat.set_beamwidth(beam_idx=beam_idx, beamwidth=beamwidth)

    self.nmc.allocate_power(dbm_power_dict)
    self.nmc.update_ues_serving_history()

  def action_to_beam_list(self, action: npt.NDArray[np.float64]) -> List[int]:
    """Map the action output to the serving beam set.

    Args:
        action (float): Action of beams

    Returns:
        Set[int]: Serving beam set.
    """
    return [i for i, power in enumerate(action) if power > 0]

  def action_to_power_dict(self, power_action: npt.NDArray[np.float64], sat_name) -> Dict[int, float]:
    """Map the action output to the beam tx power.
    Args:
        power_action (npt.NDArray[np.float64]): Action of the power
                                                of all beams and total power.
    Returns:
        Dict[int, float]: Dict of the power of each turned on beam.
    """
    online_beam = self.leo_agents[sat_name].sat.cell_topo.online_beam_set
    mw_power_dict = {}
    dbm_power_dict = {}
    for beam_idx in range(self.cell_num):
      if beam_idx in online_beam:
        mw_power_dict[beam_idx] = abs(power_action[beam_idx])
      else:
        mw_power_dict[beam_idx] = 0
    agent = self.leo_agents[sat_name]
    total_power_mw = util.tolinear(util.rescale_value(power_action[-1],
                                                      agent.total_power_low,
                                                      agent.total_power_high,
                                                      agent.sat.min_power,
                                                      agent.sat.max_power))

    multiplier = (total_power_mw
                  / (np.sum(np.fromiter(mw_power_dict.values(), dtype=float)) + constant.MIN_POSITIVE_FLOAT))
    for beam_idx in online_beam:
      mw_power_dict[beam_idx] *= multiplier

    for beam_idx in mw_power_dict:
      dbm_power_dict[beam_idx] = util.todb(mw_power_dict[beam_idx])

    return dbm_power_dict

  def action_to_beamwidth_dict(self, beamwidth_action: npt.NDArray[np.float64], sat_name) -> Dict[int, float]:
    res = {}
    agent = self.leo_agents[sat_name]
    for i, beamwidth in enumerate(beamwidth_action):
      res[i] = util.rescale_value(beamwidth,
                                  agent.beamwidth_action_low[i],
                                  agent.beamwidth_action_high[i],
                                  agent.sat.min_beamwidth,
                                  agent.sat.max_beamwidth)

    return res

  def _cal_reward(self, ue_throughput: Dict[str, float], no_action=False) -> Dict[str, float]:
    sat_tran_ratio = {}
    reward = {}
    for sat_name in self.agent_names:
      reward[sat_name] = 0

    if no_action:
      for sat_name in self.leo_agents:
        overhead = self.overflowed_overhead[sat_name]
        self.overflowed_overhead[sat_name] = max(0, self.overflowed_overhead[sat_name] - constant.MOVING_TIMESLOT)
        sat_tran_ratio[sat_name] = max(0, 1 - overhead / constant.MOVING_TIMESLOT)

    for ue_name, throughput in ue_throughput.items():
      last_satbeam = self.ue_dict[ue_name].last_serving
      # print(self.ue_dict[ue_name].servable)
      if last_satbeam is not None:
        sat_name, _ = last_satbeam
        agent = self.leo_agents[sat_name]
        if len(agent.sat.cell_topo.serving) > 0:
          if sat_name not in sat_tran_ratio:
            self.overhead[self.step_num][sat_name] = self._cal_overhead(agent)
            overhead = self.overhead[self.step_num][sat_name] + self.overflowed_overhead[sat_name]
            self.overflowed_overhead[sat_name] = max(0, overhead - constant.MOVING_TIMESLOT)
            sat_tran_ratio[sat_name] = max(0, 1 - overhead / constant.MOVING_TIMESLOT)

          self.data_rate[self.step_num][sat_name] += sat_tran_ratio[sat_name] * throughput
          self.throughput[self.step_num][sat_name] += throughput

          power_muW = (util.tolinear(agent.sat.all_power) / constant.MILLIWATT) * 1e6
          ee = (sat_tran_ratio[sat_name] * throughput / power_muW)
          self.ee[self.step_num][sat_name] += ee

          self.penalty[self.step_num][ue_name] = max((self.ue_dict[ue_name].required_datarate - throughput) / power_muW,
                                                     0)

          reward[sat_name] += max(ee - self.penalty[self.step_num][ue_name], 0)

    return reward

  def _cal_overhead(self, agent: Agent) -> float:
    pass

  def get_position_state(self, sat_name) -> npt.NDArray[np.float32]:
    return self.leo_agents[sat_name].get_scaled_pos(plot_range=self.plot_range)

  def get_state_info(self) -> Dict[str, List[float]]:
    pass

  def add_random_interference(self):
    """Other satellites have random beam act as interference """
    if self.step_num % self.interference_change_freq == 0:
      interfer_sats = set(intefere_sat_name for intefere_sat_name, _ in self.additional_beam_set)
      for intefere_sat_name in interfer_sats:
        self.constel.all_sat[intefere_sat_name].clear_power()
        self.constel.all_sat[intefere_sat_name].cell_topo.serving.clear()
      self.additional_beam_set.clear()
      for intefere_sat_name, sat in self.constel.all_sat.items():
        if self.in_range(sat, 5) and intefere_sat_name not in self.leo_agents:
          beam_set = random.sample(range(self.cell_num), self.random_interfer_beam_num)
          # print(f'intefere_sat_name: {intefere_sat_name}, beams: {beam_set}')
          for beam_idx in beam_set:
            sat.set_beam_power(beam_idx,
                               random.uniform(self.interfer_power_low, self.interfer_power_high))
            sat.cell_topo.add_serving(f'{beam_idx}', beam_idx)
            self.additional_beam_set.add((intefere_sat_name, beam_idx))

  def reset(self, seed=None, options=None) -> Tuple[Dict[str, List[float]], Any]:
    super().reset(seed=seed)

    self._init_env()
    state = self.get_state_info()
    self.reset_count += 1

    if seed:
      np.random.seed(seed)
    if options:
      print(options)
    return state, {'has_action': (self.step_num % self.action_period == 0)}

  def _init_env(self):
    self.step_num = 0
    self.constel = self.make_constellation()
    self.ues = self.make_ues()
    self.dt_server = User('DT server', position=Position(geodetic=Geodetic(longitude=constant.ORIGIN_LONG,
                                                                           latitude=constant.ORIGIN_LATI,
                                                                           height=constant.R_EARTH)), required_datarate=0)
    for ue in self.ues:
      self.ue_dict[ue.name] = ue
    self.nmc = self.make_nmc(constel=self.constel, ues=self.ues)

    for t in range(self.max_step + 1):
      self.ee[t] = {}
      self.data_rate[t] = {}
      self.overhead[t] = {}
      self.throughput[t] = {}
      self.ue_throughput[t] = {}
      self.penalty[t] = {}
      for sat_name in self.agent_names:
        self.ee[t][sat_name] = 0
        self.data_rate[t][sat_name] = 0
        self.overhead[t][sat_name] = 0
        self.throughput[t][sat_name] = 0
      for ue in self.ues:
        self.penalty[t][ue.name] = 0
        self.ue_throughput[t][ue.name] = 0

    for sat_name in self.agent_names:
      self.leo_agents[sat_name].sat = self.constel.all_sat[sat_name]
      self.leo_agents[sat_name].sat.servable = self.ues
      cell_num = self.leo_agents[sat_name].sat.cell_topo.cell_number
      self.prev_cell_sinr[sat_name] = np.zeros((cell_num, ))
      self.prev_beam_power[sat_name] = np.zeros((cell_num, ))
      self.overflowed_overhead[sat_name] = 0

    # TODO would not be feasible if cell_num is dynamic or is non-identical
    self.cell_num = self.leo_agents[self.agent_names[0]].sat.cell_topo.cell_number

  def render(self):
    sat_name = '3_0_24'
    self.ax.clear()
    util.plot_taiwan_shape(self.ax)
    for ue in self.ues:
      ue_long = ue.position.geodetic.longitude
      ue_lati = ue.position.geodetic.latitude
      self.ax.scatter(ue_long, ue_lati, s=constant.UE_MARKER_SIZE, c='g')
    for sat in self.constel.all_sat.values():
      if self.in_range(sat, self.plot_range):
        if sat.name in self.leo_agents:
          cell_plot_mode = 'active_and_training'
          self.ax.text(sat.position.geodetic.longitude,
                       sat.position.geodetic.latitude,
                       sat.name)
        else:
          cell_plot_mode = 'active_only'
        # print(f'{cell_plot_mode}')
        sat.cell_topo.plot_geodetic_cell_topology(ax=self.ax,
                                                  sat_height=sat.position.geodetic.height,
                                                  cell_range_mode='main_lobe_range',
                                                  cell_plot_mode=cell_plot_mode,
                                                  color_dict=None)

    plt.xlim((constant.ORIGIN_LONG - self.plot_range,
             constant.ORIGIN_LONG + self.plot_range))
    plt.ylim((constant.ORIGIN_LATI - self.plot_range,
             constant.ORIGIN_LATI + self.plot_range))
    self.ax.set_title(f'{self.name}    t: {self.step_num},\n'
                      f'ee: {self.ee[self.step_num][sat_name]:7.2f}, '
                      f'power: {self.leo_agents[sat_name].sat.all_power:3.2f} dBm', {'fontsize': 28})
    # self.leo_agents[sat_name].sat.cell_topo.print_all_beams()
    plt.show()
    plt.pause(0.1)

  def in_range(self, sat, r) -> bool:
    diff_la = abs(sat.position.geodetic.latitude - constant.ORIGIN_LATI)
    diff_lo = abs(sat.position.geodetic.longitude - constant.ORIGIN_LONG)
    return diff_la < r and diff_lo < r

  def make_constellation(self) -> Constellation:
    """Make a constellation"""
    shell_num = 4
    plane_num = [1, 1, 1, 1, 1]
    sat_per_plane = [22, 22, 20, 58, 43]
    sat_height = np.array([550, 540, 570, 560, 560]) * constant.KM
    inclination = np.array([53, 53.2, 70, 97.6, 97.6]) * constant.PI_IN_RAD

    constel = Constellation(setup_data_list=[
        ConstellationData(plane_num=plane_num[i],
                          sat_per_plane=sat_per_plane[i],
                          sat_height=sat_height[i],
                          inclination_angle=inclination[i])
        for i in range(shell_num)],
        args=self.args)

    # move the constellation to the desired simulated scenario
    constel.update_sat_position(constant.STARTING_TIMESLOT)
    return constel

  def load_ues_data(self):
    with open(f'low_earth_orbit/util/ue_position/ue_position_{self.args.ue_num}.csv', mode='r', newline='') as f:
      reader = csv.DictReader(f)
      for row in reader:
        ue_idx = int(row.pop('ue_idx'))
        self.ue_pos_data[ue_idx] = row

  def make_ues(self) -> List[User]:
    """Make a list of users

    Returns:
        List[User]: The user list
    """
    ues = [0] * self.args.ue_num
    for ue_idx in self.ue_pos_data:
      ues[ue_idx] = User(name=f'ue{ue_idx}',
                         position=Position(geodetic=Geodetic(longitude=float(self.ue_pos_data[ue_idx]['longitude']),
                                                             latitude=float(self.ue_pos_data[ue_idx]['latitude']),
                                                             height=float(self.ue_pos_data[ue_idx]['height']))),
                         required_datarate=self.args.R_min
                         )

    return ues

  def make_nmc(self, constel: Constellation, ues: List[User]) -> NMC:
    return NMC(constellation=constel, ues=ues)
