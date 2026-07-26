from opendbc.car import uds
from opendbc.car.carlog import carlog
from opendbc.car.isotp_parallel_query import IsoTpParallelQuery

RADAR_ADDR = 0x764
RADAR_BUS = 0


def _session_request(can_recv, can_send, session_type: int, timeout: float = 0.1) -> bool:
  request = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL, session_type])
  response = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40, session_type])
  query = IsoTpParallelQuery(can_send, can_recv, RADAR_BUS, [(RADAR_ADDR, None)], [request], [response])
  return len(query.get_data(timeout)) > 0


def enter_radar_programming_session(can_recv, can_send, retry: int = 5) -> bool:
  """Silence the radar by holding it in a UDS programming session.

  This radar does not support COMMUNICATION_CONTROL (0x28 replies NRC 0x11), so
  disable_ecu() cannot be used. A programming session stops all of its periodic frames
  (CRZ_INFO, CRZ_CTRL, 0x499, tracks 0x361-0x366) while CRZ_EVENTS and PEDALS, owned by
  other ECUs, keep transmitting. The radar stays silent as long as tester present keeps
  arriving; it falls back to the default session on the S3 timeout otherwise.
  WARNING: THIS DISABLES AEB while in effect!"""
  for i in range(retry):
    try:
      if _session_request(can_recv, can_send, uds.SESSION_TYPE.PROGRAMMING):
        carlog.warning("mazda radar programming session entered")
        return True
    except Exception:
      carlog.exception("mazda radar programming session exception")
    carlog.error(f"mazda radar programming session retry ({i + 1}) ...")

  carlog.error("mazda radar programming session failed")
  return False


def request_radar_default_session(can_recv, can_send) -> bool:
  """Return the radar to its default session, restoring stock behavior."""
  try:
    return _session_request(can_recv, can_send, uds.SESSION_TYPE.DEFAULT)
  except Exception:
    carlog.exception("mazda radar default session exception")
    return False
