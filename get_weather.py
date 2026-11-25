# only run once to asemble historycal weather data

from datetime import datetime

import pandas as pd
from meteostat import hourly

# we use LAX for the weather as it is the nearest station with reliable hourly data for that time
STATION_ID = "72295"  
OUT_CSV = "weather_la_hourly.csv"
START = datetime(2012, 3, 1)
END = datetime(2012, 6, 30, 23)
TZ = "America/Los_Angeles"

FIELDS = {"temp": "temperature", "rhum": "humidity", "prcp": "precipitation", "wspd": "wind_speed"}

raw = hourly(STATION_ID, START, END).fetch()
print(f"Raw rows returned for station {STATION_ID}: {0 if raw is None else len(raw)}")
if raw is None or raw.empty:
    raise RuntimeError(
        f"Meteostat returned no hourly data for station {STATION_ID} in this period. "
        "Check network access, or try another station ID."
    )

w = raw[list(FIELDS)].rename(columns=FIELDS)
w["wind_speed"] = w["wind_speed"] / 3.6  

w.index = w.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None)

grid = pd.date_range(START, END, freq="h")
w = w.reindex(grid)
w.index.name = "time"

missing = w.isna().sum()
print("Missing hours before fill:")
print(missing.to_string())

w = w.ffill().bfill()  
w = w.reset_index()
w.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
print(f"Wrote {OUT_CSV}: {len(w)} hourly rows")