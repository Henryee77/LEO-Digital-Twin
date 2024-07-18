"""nmc."""
from typing import List, Set

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

    self.reallocate_power(handing_sat)

  def reallocate_power(self, handing_sat: Set[str]):
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
        sat.set_beam_power(key, sat.max_power + util.todb(item / serving_num))
