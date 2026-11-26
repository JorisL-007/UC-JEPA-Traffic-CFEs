import re

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

# Use context data from same year as METR-LA measurements (closest available OSM data is from ~2-6 months later)
ox.settings.overpass_settings = (
    '[out:json][timeout:{timeout}]{maxsize}'
    '[date:"2012-09-13T00:00:00Z"]'
)
ox.settings.overpass_url = "https://overpass.private.coffee/api"
ox.settings.overpass_rate_limit = False
ox.settings.max_query_area_size = 25_000_000  # split into areas ≤25 km²
ox.settings.requests_timeout = 600           # allow 10 minutes per request
ox.settings.log_console = True               # display download progress
ox.settings.use_cache = True
ox.settings.cache_folder = "cache"

SNAP_RADIUS_M = 100
POI_BUFFER_M = 100
IN_STEPS, OUT_STEPS = 12, 12
MPH_TO_KMH = 1.609344
UTM_CRS = "EPSG:32611"


DEFAULTS = {
    "motorway": (3, 105), "motorway_link": (1, 60),
    "trunk": (3, 90), "trunk_link": (1, 60),
    "primary": (2, 70), "primary_link": (1, 50),
    "secondary": (2, 60), "secondary_link": (1, 50),
    "tertiary": (1, 50), "tertiary_link": (1, 40),
    "residential": (1, 40), "unclassified": (1, 50),
    "living_street": (1, 20), "service": (1, 30),
}
FALLBACK_DEFAULT = (1, 50)

POI_AMENITIES = ["fuel", "charging_station", "parking", "restaurant"]
POI_NAMES = ["gas_station", "charging_station", "parking", "restaurant"]


def first(v):
    return v[0] if isinstance(v, list) else v


def parse_lanes(v):
    try:
        return int(float(str(first(v)).split(";")[0]))
    except ValueError:
        return None


def parse_maxspeed(v):
    v = first(v)
    m = re.match(r"(\d+(\.\d+)?)\s*(mph)?", str(v).strip())
    if not m:
        return None
    val = float(m.group(1))
    return val * MPH_TO_KMH if m.group(3) == "mph" else val


def _decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def read_metr_la_h5(path):
    with h5py.File(path, "r") as f:
        if "df" not in f:
            raise ValueError(f"{path}: no 'df' group, so this is not a pandas fixed-format store")
        g = f["df"]
        missing = [k for k in ("axis0", "axis1", "block0_values") if k not in g]
        if missing:
            raise ValueError(f"{path}: missing datasets {missing} in the 'df' group")

        sensors = _decode(g["axis0"][:])
        items = _decode(g["block0_items"][:]) if "block0_items" in g else sensors
        times = pd.to_datetime(g["axis1"][:], unit="ns")
        values = g["block0_values"][:]

    df = pd.DataFrame(values, index=times, columns=items)
    df = df.reindex(columns=sensors)
    print(f"Loaded {path}: {df.shape[0]} timesteps x {df.shape[1]} sensors")
    return df


def build_static_features(locs):
    lats, lons = locs["latitude"], locs["longitude"]
    west, east = lons.min() - 0.01, lons.max() + 0.01
    south, north = lats.min() - 0.01, lats.max() + 0.01
    bbox = (west, south, east, north)

    # Road segments with attributes
    G = ox.graph_from_bbox(bbox=bbox, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)[["highway", "lanes", "maxspeed", "geometry"]]
    edges = edges.reset_index(drop=True).to_crs(UTM_CRS)

    # Snap sensors to the nearest road within SNAP_RADIUS_M
    sensors = gpd.GeoDataFrame(
        locs.reset_index(drop=True), geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326"
    ).to_crs(UTM_CRS)
    snapped = gpd.sjoin_nearest(
        sensors, edges, how="left", max_distance=SNAP_RADIUS_M
    )
    snapped = snapped.dropna(subset=["index_right"]).drop_duplicates("sensor_id")
    snapped = snapped.set_index("sensor_id")

    # Road features with class-based imputation
    hw = snapped["highway"].map(first).fillna("unknown")
    lanes = snapped["lanes"].map(parse_lanes)
    speed = snapped["maxspeed"].map(parse_maxspeed)
    imputed = pd.DataFrame({"lanes_imputed": lanes.isna(), "speed_imputed": speed.isna()})

    lane_default = hw.map(lambda h: DEFAULTS.get(h, FALLBACK_DEFAULT)[0])
    speed_default = hw.map(lambda h: DEFAULTS.get(h, FALLBACK_DEFAULT)[1])
    lanes = lanes.fillna(lane_default)
    speed = speed.fillna(speed_default)

    # POI counts within a buffer around each sensor
    pois = ox.features_from_bbox(bbox=bbox, tags={"amenity": POI_AMENITIES})
    pois = pois.reset_index()[["amenity", "geometry"]].to_crs(UTM_CRS)
    pois["geometry"] = pois.geometry.centroid
    buffers = snapped.geometry.buffer(POI_BUFFER_M)

    static = pd.DataFrame({"lanes": lanes, "speed_limit_kmh": speed}, index=snapped.index)
    for amenity, name in zip(POI_AMENITIES, POI_NAMES):
        pts = pois[pois["amenity"] == amenity].geometry
        static[name] = [int(pts.intersects(b).sum()) for b in buffers]

    return static, imputed


def build_dynamic_features(time_index, weather_csv):
    t = pd.DatetimeIndex(time_index)
    dow = np.eye(7)[t.day_of_week]
    hour = t.hour + t.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    w = pd.read_csv(weather_csv, parse_dates=["time"]).set_index("time").sort_index()
    w = w[["temperature", "wind_speed", "precipitation", "humidity"]]
    w = w.reindex(t, method="ffill")

    return np.column_stack([dow, hour_sin, hour_cos, w.to_numpy()]).astype(np.float32)


class METRLAContextDataset(Dataset):
    def __init__(self, speeds_kmh, dynamic, static, in_steps=IN_STEPS, out_steps=OUT_STEPS):
        self.mask = (~np.isnan(speeds_kmh)).astype(np.float32)
        self.speeds = np.nan_to_num(speeds_kmh, nan=0.0).astype(np.float32)
        self.dynamic = dynamic          
        self.static = static.astype(np.float32)  
        self.in_steps, self.out_steps = in_steps, out_steps
        self.T, self.N = speeds_kmh.shape
        self.D = dynamic.shape[1]
        self.S = static.shape[1]

    def __len__(self):
        return self.T - self.in_steps - self.out_steps + 1

    def __getitem__(self, i):
        a = i
        b = i + self.in_steps
        c = b + self.out_steps

        speed_in = self.speeds[a:b, :, None]
        dyn = np.broadcast_to(self.dynamic[a:b, None, :], (self.in_steps, self.N, self.D))
        stat = np.broadcast_to(self.static[None], (self.in_steps, self.N, self.S))
        x = np.concatenate([speed_in, dyn, stat], axis=-1)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(self.speeds[b:c]),
            "y_mask": torch.from_numpy(self.mask[b:c]),
        }


if __name__ == "__main__":
    # Inputs
    speeds_mph = read_metr_la_h5("data/METR-LA.h5")
    speeds_mph.columns = speeds_mph.columns.astype(str)

    locs = pd.read_csv("data/graph_sensor_locations.csv")
    locs["sensor_id"] = locs["sensor_id"].astype(str)
    locs = locs.set_index("sensor_id", drop=False)

    # Static features and sensor matching
    static_df, imputed = build_static_features(locs)
    keep = [c for c in speeds_mph.columns if c in static_df.index]
    print(f"Sensors kept after snapping: {len(keep)} of {speeds_mph.shape[1]}")
    print(f"Imputed lanes: {imputed.loc[keep, 'lanes_imputed'].sum()}, "
          f"imputed speed limits: {imputed.loc[keep, 'speed_imputed'].sum()}")

    # Speeds in km/h (zeros are missing readings in METR-LA)
    speeds_kmh = speeds_mph[keep].replace(0, np.nan).to_numpy() * MPH_TO_KMH
    static = static_df.loc[keep].to_numpy()

    # Dynamic features
    dynamic = build_dynamic_features(speeds_mph.index, "data/weather_la_hourly.csv")

    dataset = METRLAContextDataset(speeds_kmh, dynamic, static)
    sample = dataset[0]
    print(len(dataset), "samples")
    print({k: tuple(v.shape) for k, v in sample.items()})

    np.savez_compressed(
        "data/metr_la_context.npz",
        speeds_kmh=speeds_kmh,
        dynamic=dynamic,
        static=static,
        sensor_ids=np.asarray(keep, dtype=str),
        timestamps=speeds_mph.index.to_numpy(),
        static_feature_names=static_df.columns.to_numpy(dtype=str),
        lanes_imputed=imputed.loc[keep, "lanes_imputed"].to_numpy(),
        speed_imputed=imputed.loc[keep, "speed_imputed"].to_numpy(),
        osm_settings=ox.settings.overpass_settings,
    )
    print("Saved data/metr_la_context.npz")