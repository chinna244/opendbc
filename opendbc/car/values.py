from typing import get_args
from opendbc.car.body.values import CAR as BODY
from opendbc.car.chrysler.values import CAR as CHRYSLER
from opendbc.car.ford.values import CAR as FORD
from opendbc.car.gm.values import CAR as GM
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.mazda.values import CAR as MAZDA
from opendbc.car.mock.values import CAR as MOCK
from opendbc.car.nissan.values import CAR as NISSAN
from opendbc.car.psa.values import CAR as PSA
from opendbc.car.rivian.values import CAR as RIVIAN
from opendbc.car.mg.values import CAR as MG
from opendbc.car.subaru.values import CAR as SUBARU
from opendbc.car.tesla.values import CAR as TESLA
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.volkswagen.values import CAR as VOLKSWAGEN

Platform = BODY | CHRYSLER | FORD | GM | HONDA | HYUNDAI | MAZDA | MOCK | MG | NISSAN | PSA | RIVIAN | SUBARU | TESLA | TOYOTA | VOLKSWAGEN
BRANDS = get_args(Platform)

PLATFORMS: dict[str, Platform] = {str(platform): platform for brand in BRANDS for platform in brand}


def platform_from_vin(vin: str) -> str | None:
  """The one platform the VIN's fields identify, or None when the VIN is unknown to every
  platform or ambiguous.

  Platforms advertise VIN metadata (wmis, chassis_codes, years) on their config; those
  without any are skipped. A hardware swap cannot change the VIN, so this is the strongest
  signal to check a user-selected platform bundle against.
  """
  from opendbc.car.vin import Vin, is_valid_vin

  if not is_valid_vin(vin):
    return None

  vin_obj = Vin(vin)
  chassis_code = vin_obj.vds[0:2]
  year = vin_obj.vis[0]

  candidates = set()
  for platform in PLATFORMS.values():
    config = platform.config
    wmis = getattr(config, 'wmis', None)
    if not wmis:
      continue
    if vin_obj.wmi in wmis and chassis_code in getattr(config, 'chassis_codes', set()) \
       and year in getattr(config, 'years', set()):
      candidates.add(str(platform))

  return next(iter(candidates)) if len(candidates) == 1 else None
