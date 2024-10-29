"""The Ground User module."""

from typing import Dict, Tuple, List
import collections

from ..util import Position
from ..util import constant
from ..util import UEservable, RxData, SatBeamID


class User(object):
  """The user class.

  Attributes:
    servable: The dict that Satellite can served.
    serving_history: The deque that save the serving hitsory
    a3_table: The a3 trigger time table
    serving_sinr: The serving SINR
    online: Is this UE is been serving or not
  """
  servable: UEservable
  serving_history: collections.deque[Tuple[SatBeamID, Position, float]]
  a3_table: Dict[SatBeamID, int]
  serving_sinr: float = constant.MIN_NEG_FLOAT
  _online: bool = False

  # Need to check if the user didn't com for a long time

  def __init__(
      self,
      name: str,
      position: Position,
      rx_gain: float = constant.DEFAULT_RX_GAIN,
      training_window_size: int = constant.DEFAULT_TRAINING_WINDOW_SIZE
  ):
    """The __init__ funciton for user.

    Args:
      name (str): The name of the object.
      position (position): The position of the object.
      rx_gain (float): The recieving antenna gain of the object

    Raise:
      ValueError: The user must initialize the position
    """
    self.name = name
    self.position = position
    self.rx_gain = rx_gain
    self.__training_window_size = training_window_size

    self.servable = {}
    self.serving_history = collections.deque(maxlen=training_window_size)
    self.a3_table = {}

  @property
  def training_window_size(self):
    return self.__training_window_size

  @property
  def last_serving_history(self):
    if len(self.serving_history) == 0:
      return None
    return self.serving_history[-1]

  @property
  def last_serving(self) -> SatBeamID:
    """The last serving SatBeamID.

    Returns:
        SatBeamID: sat name and beam index. (tuple(str, int))
    """
    if not self.online:
      return None
    if self.serving_history:
      return self.last_serving_history[0]
    return None

  @property
  def time_to_trigger(self):
    if self.online:
      return constant.TIME_TO_TRIGGER
    return constant.OFFLINE_TIME_TO_TRIGGER

  @property
  def online(self):
    return self._online

  @property
  def data_rate(self) -> float:
    return constant.DEFAULT_GATEWAY_THROUGHPUT

  @online.setter
  def online(self, status: bool):
    self._online = status
    if not status:
      self.serving_sinr = constant.MIN_NEG_FLOAT

  def trans_latency(self, data_size: int) -> float:
    """Transmission latency

    Args:
        data_size (int): byte

    Returns:
        float: latency
    """
    return data_size / self.data_rate

  def servable_clear(self) -> None:
    """Clear the servable dict."""
    self.servable.clear()

  def servable_add(self, name_sat: str, beam_num: int, rsrp: float) -> None:
    """Add servable sat information to UE.

    Args:
        name_sat (str): The name of the sat
        beam_num (int): The beam number of the sat
        rsrp (float): The RSRP value for scaning.
    """
    rx_data = RxData(rsrp=rsrp)
    self.servable[(name_sat, beam_num)] = rx_data

  def filter_servable(self, filter_target: List[SatBeamID]):
    new_servable = {}
    for sat_beam in filter_target:
      if sat_beam in self.servable:
        new_servable[sat_beam] = self.servable[sat_beam]

    self.servable = new_servable

  def serving_add(self, sat_beam: SatBeamID, beam_pos=None, serv_sinr=None) -> None:
    """Add final result for the serving.

    Args:
        name_sat (str): The name of the server
        beam_num (int): The bean index of sat
    """
    self.online = True
    self.serving_history.append((sat_beam, beam_pos, serv_sinr))

    if serv_sinr is None:
      self.serving_sinr = self.servable[sat_beam].rsrp
    else:
      self.serving_sinr = serv_sinr

  def update_one_beam_a3(self, sat_beam: SatBeamID) -> int:
    """Get the time of a3 event for sat_beam.

    Args:
        sat_beam (SatBeamID): The sat_beam to check in the a3_table.

    Returns:
        (int): The update time for sat_beam.
    """
    if sat_beam in self.a3_table:
      return self.a3_table[sat_beam] + constant.A3_ONE_TIMESLOT
    return constant.A3_ONE_TIMESLOT

  def update_a3_table(self, threshold: float) -> None:
    """Update the A3 table with threshold.

    Args:
        threshold (float): The threshold for the servable in A3 event.
    """
    new_a3_table: Dict[SatBeamID, int] = {}
    for key, data in self.servable.items():
      if data.gamma > threshold:
        new_a3_table[key] = self.update_one_beam_a3(key)

    self.a3_table = new_a3_table

  def get_a3_target(self) -> SatBeamID | None:
    """Get the handoff target based on a3 table.

    Returns:
      (SatBeamID | None): The handoff target, which might be None.
    """
    max_index = None
    max_value = constant.MIN_NEG_FLOAT

    time_threshold = self.time_to_trigger

    for key, item in self.a3_table.items():
      rsrp = self.servable[key].rsrp
      if item > time_threshold and rsrp > max_value:
        max_index = key
        max_value = rsrp

    if max_index is not None:
      self.a3_table[max_index] = constant.A3_RESET

    return max_index

  def get_next_sat(self) -> SatBeamID | None:
    """Get the next sat based on the A3 table.

    Returns:
        SatBeamID | None: The next SatBeamID, which might be None.
    """
    a3_target = self.get_a3_target()

    if not a3_target and self.last_serving in self.servable:
      return self.last_serving

    return a3_target

  def a3_event_check(self) -> Tuple[bool, SatBeamID | None, SatBeamID | None]:
    """A3 event entry point

    Returns:
        Tuple[bool, SatBeamID | None]: If handoff and the the next SatBeamID
    """
    handoff = False
    next_serving = None

    if self.servable:
      threshold = self.serving_sinr
      self.update_a3_table(threshold)
      next_serving = self.get_next_sat()

    if self.last_serving != next_serving:
      handoff = True
      '''print(f'Handoff UE: {self.name}, From: {self.last_serving}, '
            f'To: {next_serving}')'''

    return handoff, self.last_serving, next_serving
