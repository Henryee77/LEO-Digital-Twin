"""nmc."""
from typing import List, Set, Dict

from ..ground_user import User
from ..constellation import Constellation
from ..util import util
from ..util import SatBeamID


class NMC(object):

  def __init__(self, constellation: Constellation, ues: List[User]):

    self.constellation = constellation
    self.ues = ues

  @property
  def ue_num(self):
    return len(self.ues)

  def handoff(self, ue: User, next_serving: SatBeamID | None):
    """The handoff action in the NMC.

    Args:
        ue (User): The user
        next_serving (SatBeamID | None): The next sat.
    """
    if ue.last_serving:
      (sat_name, _) = ue.last_serving
      self.constellation.drop_serving(sat_name, ue.name)

    if next_serving:
      self.constellation.add_serving(next_serving, ue.name)

  def a3_event_check(self):
    """A3 event check entry point."""
    handing_sat = set()
    for ue in self.ues:
      handoff, last_serving, next_serving = ue.a3_event_check()
      # print(ue.name, handoff, last_serving, next_serving)
      # print(ue.servable)
      if handoff:
        self.handoff(ue, next_serving)
        if last_serving:
          sat_name, _ = last_serving
          handing_sat.add(sat_name)

      if next_serving:
        ue.serving_add(next_serving)
        sat_name, _ = next_serving
        handing_sat.add(sat_name)
      else:
        ue.online = False

    self.equally_allocate_power(handing_sat)

  def allocate_power(self, power_dict: Dict[str, Dict[int, float]]):
    for sat_name in power_dict:
      self.constellation.all_sat[sat_name].clear_power()
      for beam_idx, dbm_power in power_dict[sat_name].items():
        self.constellation.all_sat[sat_name].set_beam_power(beam_idx=beam_idx, tx_power=dbm_power)

  def update_ues_serving_history(self):
    sinr_dict = self.constellation.cal_transmission_sinr(ues=self.ues)
    for ue in self.ues:
      satbeam = ue.last_serving
      if satbeam:
        beampos = self.constellation.all_sat[satbeam[0]].cell_topo.beam_list[satbeam[-1]].center_point
        serv_sinr = sinr_dict[ue.name]
        ue.serving_history[-1] = (satbeam, beampos, serv_sinr)

  def equally_allocate_power(self, handing_sat: Set[str]):
    """Allocate the power of the satellite
       that has handover or handoff

    Args:
        handing_sat (Set[str]): The Set of satellite name that their power
                                need to be allocated
    """
    for sat_name in handing_sat:
      sat = self.constellation.all_sat[sat_name]
      serving_num = len(sat.cell_topo.serving)
      sat.clear_power()
      for key, item in sat.cell_topo.serving_status.items():
        # print(f'sat_name:, {sat_name}, serving_num: {serving_num}, key: {key}, item: {item}, power_ratio: {util.todb(item / serving_num)}')
        sat.set_beam_power(key, sat.max_power + util.todb(item / serving_num))

      # print('nmc:')
      # sat.cell_topo.print_all_beams()
