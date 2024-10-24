"""leosat_env.py"""
from __future__ import annotations
from typing import List, Set, Dict, Tuple, Any
import random
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from gymnasium import spaces
from low_earth_orbit.ground_user import User
from low_earth_orbit.constellation import Constellation
from low_earth_orbit.constellation import ConstellationData
from low_earth_orbit.nmc import NMC
from low_earth_orbit.channel import Channel
from low_earth_orbit.util import util
from low_earth_orbit.util import Position
from low_earth_orbit.util import Geodetic
from low_earth_orbit.util import constant
from MA_TD3.agent.agent import Agent


class LEOSatEnv(gym.Env):
  """The LEO Env class."""

  def __init__(self,
               ax: plt.Axes,
               args,
               agent_dict: Dict[str, Agent],
               real_agents: Dict[str, Agent],
               digital_agents: Dict[str, Agent],
               agent_names: List[str]):
    super(LEOSatEnv, self).__init__()
    self.name = 'LEOSat'
    self.ax = ax
    # self.main_sat_name = '0_3_8'
    self.constel = self.ues = self.nmc = None
    self.ue_dict = {}
    self.prev_cell_sinr = {}
    self.prev_beam_power = {}
    self.reward = {}
    self._leo_agents = agent_dict
    self.real_agents = real_agents
    self.digital_agents = digital_agents
    self.agent_num = len(agent_names)
    self.agent_names = agent_names
    self.cell_num = 0
    if self.leo_agents is not None:
      self._init_env()

    self.step_num = 0
    self.max_step = args.step_per_ep
    self.plot_range = 2  # the plotting range

    self.action_space = spaces.Box(np.array([-1]), np.array([1]))  # dummy for gym template
    self.observation_space = spaces.Box(np.array([-10]), np.array([10]))

    self.max_power = constant.MAX_POWER

    self.additional_beam_set = set()
    self.random_interfer_beam_num = 2
    self.interference_change_freq = 5
    self.interfer_power_low = 20
    self.interfer_power_high = util.truncate(
      self.max_power - util.todb(self.random_interfer_beam_num))

  @ property
  def name(self):
    return self._name

  @ name.setter
  def name(self, name):
    self._name = name

  @ property
  def leo_agents(self) -> Dict[str, Agent]:
    return self._leo_agents

  @ leo_agents.setter
  def leo_agents(self, agent_dict: Dict[str, Agent]):
    self._leo_agents = agent_dict

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

  # def step_constellation_movement(self):

  def step(self, action_n: Dict[str, npt.NDArray]):
    # moving satellites
    self.constel.update_sat_position()

    for ue in self.ues:
      ue.servable_clear()

    for sat_name, action in action_n.items():
      # print(f'{sat_name} action: {action}')
      self._take_action(action, sat_name)

    self.nmc.a3_event_check()
    # self.add_random_interference()

    for sat_name, action in action_n.items():
      # set the beam power of the main satellite
      power_dict = self.action_to_power_dict(action[self.leo_agents[sat_name].power_slice], sat_name)
      self.leo_agents[sat_name].sat.clear_power()
      for beam_idx, power in power_dict.items():
        self.leo_agents[sat_name].sat.set_beam_power(beam_idx=beam_idx, tx_power=power)

    ue_sinr = self.constel.cal_transmission_sinr(ues=self.ues,
                                                 interference_beams=self.additional_beam_set)
    ue_throughput = self.constel.cal_throughput(ues=self.ues,
                                                sinr=ue_sinr,
                                                interference_beams=self.additional_beam_set)

    cell_sinr = {}
    for sat_name in self.agent_names:
      cell_sinr[sat_name] = np.zeros((self.cell_num, ))
    for ue in self.ues:
      if ue.last_serving is not None:
        sat_name, beam_idx = ue.last_serving
        cell_sinr[sat_name][beam_idx] += util.tolinear(ue_sinr[ue.name])
    for sat_name in self.agent_names:
      for beam_idx, user_num in self.leo_agents[sat_name].sat.cell_topo.serving_status.items():
        cell_sinr[sat_name][beam_idx] = util.todb(cell_sinr[sat_name][beam_idx] / user_num)

    beam_power = {}
    for sat_name in self.agent_names:
      tmp_beam_power = np.zeros((self.cell_num, ))
      sat = self.leo_agents[sat_name].sat
      for beam_idx in range(self.cell_num):
        tmp_beam_power[beam_idx] = sat.get_beam_info(beam_idx)[0]
      beam_power[sat_name] = tmp_beam_power

    self._cal_reward(ue_throughput=ue_throughput)

    self.step_num += 1
    done = (self.step_num >= self.max_step)
    truncated = (self.step_num >= self.max_step)

    obs = self.get_state_info(cell_sinr, beam_power)
    return (obs, self.reward, done, truncated, {})

  def _cal_reward(self, ue_throughput):
    for key in self.reward:
      self.reward[key] = 0.0
    # print(f'dB: {sat_power}')
    # print(util.tolinear(sat_power))
    sat_tran_ratio = {}
    for ue_name, throughput in ue_throughput.items():
      last_satbeam = self.ue_dict[ue_name].last_serving
      # print(self.ue_dict[ue_name].servable)
      if last_satbeam is not None:
        sat_name, _ = last_satbeam
        agent = self.leo_agents[sat_name]
        if len(agent.sat.cell_topo.serving) > 0:
          if sat_name not in sat_tran_ratio:
            dt_comp_latency = max([dt.computation_latency for dt in self.digital_agents.values()])
            leo2dt_distance = self.dt_server.position.calculate_distance(agent.sat.position)

            overhead = (max(agent.sat.beam_training_latency,
                            (util.rt_delay(len(self.leo_agents) * len(self.ues))
                             + self.dt_server.trans_latency(agent.state_dim * constant.INT_SIZE)
                             + util.propagation_delay(leo2dt_distance)))
                        + agent.computation_latency + agent.sat.trans_latency(len(self.ues) * constant.FLOAT_SIZE, self.dt_server) + util.propagation_delay(leo2dt_distance))
            # print(agent.sat.beam_training_latency, agent.computation_latency)
            sat_tran_ratio[sat_name] = max(0, 1 - overhead / constant.TIMESLOT)

          self.reward[sat_name] += sat_tran_ratio[sat_name] * throughput / util.tolinear(agent.sat.all_power) / 1e3

  def get_position_state(self, sat_name) -> npt.NDArray[np.float32]:
    return self.leo_agents[sat_name].get_scaled_pos(plot_range=self.plot_range)

  def get_state_info(self, cell_sinr, beam_power, init=False) -> Dict[str, List[float]]:
    pass

  def _take_action(self):
    """Take action"""
    raise ValueError('At template funtion')

  def action_to_power_dict(self, power_action: npt.NDArray[np.float64], sat_name) -> Dict[int, float]:
    raise ValueError('At template funtion')

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

  def action_to_trainset(self, action: npt.NDArray[np.float64]) -> Set[int]:
    """Map the action output to the training beam set.

    Args:
        action (float): Action

    Returns:
        Set[int]: Training beam set.
    """
    train_set = set()
    # print(action)
    for i, act in enumerate(action):
      if act >= 0:
        train_set.add(i)
    # print(action)
    # print(train_set)
    return train_set

  def reset(self, seed=None, options=None) -> Tuple[Dict[str, List[float]], Any]:
    super().reset(seed=seed)

    self._init_env()
    state = self.get_state_info(self.prev_cell_sinr, self.prev_beam_power, init=True)

    if seed:
      np.random.seed(seed)
    if options:
      print(options)
    return state, {}

  def _init_env(self):
    self.step_num = 0
    self.wireless_channel = Channel()
    self.constel = self.make_constellation(channel=self.wireless_channel)
    self.ues = self.make_ues()
    self.dt_server = User('DT server', position=Position(geodetic=Geodetic(longitude=constant.ORIGIN_LONG,
                                                                           latitude=constant.ORIGIN_LATI,
                                                                           height=constant.R_EARTH)))
    for ue in self.ues:
      self.ue_dict[ue.name] = ue
    self.nmc = self.make_nmc(constel=self.constel, ues=self.ues)

    for sat_name in self.agent_names:
      self.leo_agents[sat_name].sat = self.constel.all_sat[sat_name]
      self.leo_agents[sat_name].sat.servable = self.ues
      cell_num = self.leo_agents[sat_name].sat.cell_topo.cell_number
      self.prev_cell_sinr[sat_name] = np.zeros((cell_num, ))
      self.prev_beam_power[sat_name] = np.zeros((cell_num, ))
      self.reward[sat_name] = 0.0

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
                      f'reward: {self.reward[sat_name]:7.2f}, '
                      f'power: {self.leo_agents[sat_name].sat.all_power:3.2f} dBm', {'fontsize': 28})
    # self.leo_agents[sat_name].sat.cell_topo.print_all_beams()
    plt.show()
    plt.pause(0.1)

  def in_range(self, sat, r) -> bool:
    diff_la = abs(sat.position.geodetic.latitude - constant.ORIGIN_LATI)
    diff_lo = abs(sat.position.geodetic.longitude - constant.ORIGIN_LONG)
    return diff_la < r and diff_lo < r

  def make_constellation(self, channel: Channel) -> Constellation:
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
        channel=channel)

    # move the constellation to the desired simulated scenario
    constel.update_sat_position(-20 * constant.TIMESLOT)
    return constel

  def make_ues(self) -> List[User]:
    """Make a list of users

    Returns:
        List[User]: The user list
    """
    ue_long = [120.99, 121.58, 120.5]
    ue_lati = [24.78, 25.03, 22.33]
    if len(ue_long) != len(ue_lati):
      raise ValueError('The length of ue_long and ue_lati is not the same.')
    ues = [User(f'ue{i}',
                Position(geodetic=Geodetic(
                  longitude=ue_long[i],
                  latitude=ue_lati[i],
                  height=constant.R_EARTH))) for i in range(len(ue_long))]

    return ues

  def make_nmc(self, constel: Constellation, ues: List[User]) -> NMC:
    return NMC(constellation=constel, ues=ues)
