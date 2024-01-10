import asyncio
import datetime

from pyairnow import WebServiceAPI


async def main() -> None:
  client = WebServiceAPI('your-api-key')

  # Get current observational data based on a zip code
  data = await client.observations.zipCode(
    '90001',
    # if there are no observation stations in this zip code, optionally
    # provide a radius to search (in miles)
    distance=50,
  )

  # Get current observational data based on a latitude and longitude
  data = await client.observations.latLong(
    34.053718, -118.244842,
    # if there are no observation stations at this location, optionally
    # provide a radius to search (in miles)
    distance=50,
  )

  # Get forecast data based on a zip code
  data = await client.forecast.zipCode(
    '90001',
    # to get a forecast for a certain day, provide a date in yyyy-mm-dd,
    # if not specified the current day will be used
    date='2020-09-01',
    # if there are no observation stations in this zip code, optionally
    # provide a radius to search (in miles)
    distance=50,
  )

  # Get forecast data based on a latitude and longitude
  data = await client.forecast.latLong(
    # Lat/Long may be strings or floats
    '34.053718', '-118.244842',
    # forecast dates may also be datetime.date or datetime.datetime objects
    date=datetime.date(2020, 9, 1),
    # if there are no observation stations in this zip code, optionally
    # provide a radius to search (in miles)
    distance=50,
  )


asyncio.run(main())