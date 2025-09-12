from __future__ import annotations
import math
from dataclasses import dataclass
from multiprocessing.resource_sharer import stop
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Geo & Temporal Helpers
# =========================

R_EARTH = 6371000.0  # m

def _to_datetime_utc(series: pd.Series) -> pd.Series:
    """Converts to UTC datetime (naive->UTC)."""
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out

def _sec_diff(timestamps: pd.Series) -> np.ndarray:
    """Returns time deltas (s) between successive samples."""
    dt = (timestamps.values[1:].astype('datetime64[ns]').astype(np.int64) -
          timestamps.values[:-1].astype('datetime64[ns]').astype(np.int64)) / 1e9
    return dt

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in meters between two points (deg)."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R_EARTH * math.asin(math.sqrt(a))

def _latlon_to_local_xy(lat, lon, lat0, lon0):
    """Approx ENU 2D (x East, y North) in meters, centered on (lat0, lon0). Sufficient for ranges ~a few tens of km."""
    lat_rad = np.radians(lat)
    lat0_rad = math.radians(lat0)
    x = (np.radians(lon - lon0) * np.cos(lat0_rad)) * R_EARTH
    y = (lat_rad - lat0_rad) * R_EARTH
    return x, y

def _local_xy_to_latlon(x, y, lat0, lon0):
    lat0_rad = math.radians(lat0)
    lat = np.degrees(y / R_EARTH + lat0_rad)
    lon = np.degrees(x / (R_EARTH * np.cos(lat0_rad))) + lon0
    return lat, lon

def _ground_speed_mps(df: pd.DataFrame) -> np.ndarray:
    """Ground horizontal speed (m/s) via velN/velE if available, otherwise via lat/lon derivative."""
    if {"velN", "velE"}.issubset(df.columns):
        v = np.sqrt(df["velN"].to_numpy(float)**2 + df["velE"].to_numpy(float)**2)
        return v
    # simple fallback via distance/dt
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    time = _to_datetime_utc(df["time"])
    dt = np.r_[np.nan, _sec_diff(time)]
    d = np.r_[np.nan, [_haversine_m(lat[i-1], lon[i-1], lat[i], lon[i]) for i in range(1, len(df))]]
    v = d / dt
    return v

def _vertical_speed_up_mps(df: pd.DataFrame) -> np.ndarray:
    """Vertical speed (m/s, positive upwards). If velD (Down, NED) exists, vz = -velD, otherwise altitude derivative."""
    if "velD" in df.columns:
        return -df["velD"].to_numpy(float)
    # fallback via altitude derivative hMSL
    time = _to_datetime_utc(df["time"])
    dt = np.r_[np.nan, _sec_diff(time)]
    h = df["hMSL"].astype(float).to_numpy()
    vz = np.r_[np.nan, np.diff(h) / dt]
    return vz

def _rolling_sustain(mask: np.ndarray, min_samples: int) -> np.ndarray:
    """True if the condition is true over a sliding window of size min_samples."""
    # logical convolution (via sum)
    arr = mask.astype(int)
    kernel = np.ones(min_samples, dtype=int)
    roll = np.convolve(arr, kernel, mode="same")
    return roll >= min_samples

def _mode_dt_seconds(dt: np.ndarray, bin_res: float = 0.02) -> float:
    """Mode of time steps (>0) by histogram (resolution bin_res s)."""
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    bins = int(np.ceil((dt.max() - dt.min()) / bin_res)) + 1
    hist, edges = np.histogram(dt, bins=bins)
    i = np.argmax(hist)
    # bin center
    return float((edges[i] + edges[i+1]) / 2)

def _noise_like(residual: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Gaussian noise based on the standard deviation of high frequencies of the residual."""
    res = residual[np.isfinite(residual)]
    if res.size < 8:
        sigma = 0.0
    else:
        sigma = np.std(res - pd.Series(res).rolling(11, min_periods=1, center=True).mean().to_numpy())
    if sigma == 0:
        return np.zeros_like(residual)
    rng = np.random.default_rng(42)  # deterministic for reproducibility (change if needed)
    out = rng.normal(0.0, scale * sigma, size=residual.shape)
    return out


# =========================
# 1) Read CSV (2 header lines)
# =========================

def read_csv_gps(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Reads a CSV with 2 header lines: 1) names  2) units.
    Returns (df, units). Expected standard columns: time, lat, lon, hMSL, velN, velE, velD...
    """
    # Read the first 2 lines to get names + units
    with open(path, "r", encoding="utf-8") as f:
        header_names = f.readline().strip().split(",")
        header_units = f.readline().strip().split(",")
    names = [h.strip() for h in header_names]
    units = {}
    for n, u in zip(names, header_units):
        u = u.strip()
        u = u.replace("(", "").replace(")", "") if u else ""
        units[n] = u

    df = pd.read_csv(path, skiprows=2, names=names)
    if "time" in df.columns:
        df["time"] = _to_datetime_utc(df["time"])
    # Cast numeric columns
    for c in ["lat", "lon", "hMSL", "velN", "velE", "velD", "heading"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, units


# =========================
# 2) Sampling Frequency
# =========================

def determine_frequency(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Estimates the sampling frequency (Hz) and nominal time step (s) using the mode of deltas.
    Handles data gaps.
    """
    t = _to_datetime_utc(df["time"])
    dt = _sec_diff(t)
    dt_mode = _mode_dt_seconds(dt, bin_res=0.02)
    if not np.isfinite(dt_mode) or dt_mode <= 0:
        return np.nan, np.nan
    f = 1.0 / dt_mode
    return f, dt_mode


# =========================
# 3) Start of Aircraft Ascent
# =========================

def start_aircraft_ascent(df: pd.DataFrame, speed_threshold_kmh: float = 30.0, sustain_s: float = 5.0) -> Optional[int]:
    """
    Detects the first moment when horizontal speed exceeds speed_threshold_kmh and sustains for sustain_s.
    Returns the index (row) or None if not found.
    """
    v = _ground_speed_mps(df)
    t = _to_datetime_utc(df["time"])
    f, dt_mode = determine_frequency(df)
    if not np.isfinite(dt_mode) or dt_mode <= 0:
        # Fallback: 1 s
        dt_mode = 1.0
    win = max(1, int(round(sustain_s / dt_mode)))
    mask = (v * 3.6) > speed_threshold_kmh
    sustain = _rolling_sustain(mask, win)
    idx = np.argmax(sustain) if sustain.any() else None
    if idx is None or sustain[idx] == 0:
        return None
    return int(idx)


# =========================
# 4) Ground Altitude (Terrain) & AGL
# =========================

def average_ground_altitude(df: pd.DataFrame, idx_ascent: Optional[int]) -> float:
    """
    Estimates the terrain altitude (m MSL) as the average of hMSL before the aircraft ascent.
    If idx_ascent is None, averages over the first 5 minutes.
    """
    h = df["hMSL"].to_numpy(float)
    t = _to_datetime_utc(df["time"])
    if idx_ascent is not None and idx_ascent > 5:
        h0 = h[:idx_ascent]
    else:
        # 5 minutes if possible
        if t.notna().any():
            t0 = t.iloc[0]
            mask = (t - t0).dt.total_seconds() <= 300
            h0 = h[mask.to_numpy()]
        else:
            h0 = h[:max(1, int(len(h) * 0.1))]
    return float(np.nanmean(h0))

def agl_series(df: pd.DataFrame, ground_altitude_msl: float) -> np.ndarray:
    """
    Altitude AGL = hMSL - ground_altitude_msl.
    """
    return df["hMSL"].to_numpy(float) - float(ground_altitude_msl)


# =========================
# 5) Drop / vertical speed gain
# =========================

def start_drop(df: pd.DataFrame,
               agl: np.ndarray,
               vz_desc_threshold_mps: float = 5.0,
               min_duration_s: float = 1.5,
               agl_min_m: float = 3000.0) -> Optional[int]:
    """
    Detects the start of the 'drop/vertical speed gain' phase:
      - AGL >= agl_min_m
      - Vertical speed 'downwards' > vz_desc_threshold_mps (i.e., vz_up < -threshold)
      - Condition sustained for min_duration_s
    Returns the index or None.
    """
    vz_up = _vertical_speed_up_mps(df)  # >0 upwards
    # "rapid descent" condition
    cond = (agl >= agl_min_m) & (vz_up < -abs(vz_desc_threshold_mps))
    f, dt_mode = determine_frequency(df)
    if not np.isfinite(dt_mode) or dt_mode <= 0:
        dt_mode = 1.0
    win = max(1, int(round(min_duration_s / dt_mode)))
    sustain = _rolling_sustain(cond, win)
    idx = np.argmax(sustain) if sustain.any() else None
    if idx is None or sustain[idx] == 0:
        return None
    return int(idx)


# =========================
# 6) Trajectory reference point (drop + 9 s)
# =========================

def reference_point(df: pd.DataFrame, idx_drop: int=0, vhTreshold : float= 10.0, offset_s: float = 9.0) -> Optional[int]:
    """
    Returns the index closest to (10m/s after drop + offset_s).
    """
    #search for index after idx_drop where vertical speed > vhTreshold
    vz_up = _vertical_speed_up_mps(df)
    idx_vh = None
    for i in range(idx_drop + 1, len(vz_up)):
        if vz_up[i] > vhTreshold:
            idx_vh = i
            break
    if idx_vh is None:
        idx_vh = idx_drop
    t = _to_datetime_utc(df["time"])
    t_ref = t.iloc[idx_vh] + pd.to_timedelta(offset_s, unit="s")
    # index of the closest time
    i = int(np.argmin(np.abs((t - t_ref).dt.total_seconds().to_numpy())))
    return i

# =========================
# 7) Start of glide (2500 m) & end (1500 m)
# =========================
def agl_index(df: pd.DataFrame, agl: np.ndarray, idx_start_search: Optional[int], threshold=1000) -> Optional[int]:
    """
    First passage below startagl m AGL after idx_start_search (inclusive).
    """
    start = idx_start_search or 0
    idxs = np.where(agl[start:] < threshold)[0]
    return int(start + idxs[0]) if idxs.size else None

def start_glide_2500(df: pd.DataFrame, agl: np.ndarray, idx_start_search: Optional[int], threshold=2500) -> Optional[int]:
    """
    First passage below 2500 m AGL after idx_start_search (inclusive).
    """
    start = idx_start_search or 0
    idxs = np.where(agl[start:] < threshold)[0]
    return int(start + idxs[0]) if idxs.size else None

def end_glide_1500(df: pd.DataFrame, agl: np.ndarray, idx_glide_start: Optional[int], threshold=1500) -> Optional[int]:
    """
    First passage below 1500 m AGL after the start of the glide.
    """
    if idx_glide_start is None:
        return None
    idxs = np.where(agl[idx_glide_start:] < threshold)[0]
    return int(idx_glide_start + idxs[0]) if idxs.size else None


# =========================
# 8) Parachute deployment (1500 -> 1100 m) by sudden deceleration
# =========================

def parachute_deployment(df: pd.DataFrame,
                         agl: np.ndarray,
                         min_agl_m: float = 500.0,
                         max_agl_m: float = 1500.0,
                         window_s: float = 2.0,
                         min_drop_kmh: float = 20.0,
                         startidx: Optional[int] = None) -> Optional[int]:
    """
    Searches, between 1500 and 1100 m AGL, for the moment when ground speed undergoes the sharpest drop
    (> min_drop_kmh in window_s). Returns the index of the start of the drop.
    """
    v = _ground_speed_mps(df) * 3.6  # km/h
    t = _to_datetime_utc(df["time"])
    f, dt_mode = determine_frequency(df)
    if not np.isfinite(dt_mode) or dt_mode <= 0:
        dt_mode = 1.0
    win = max(1, int(round(window_s / dt_mode)))

    zone = (agl <= max_agl_m) & (agl >= min_agl_m) & np.isfinite(v)
    idx = np.where(zone)[0]
    if idx.size < win + 1:
        return None

    # Calculate the local drop v[i] - v[i+win]
    best_i = None
    best_drop = 0.0
    for i in idx[:-win]:
        drop = v[i] - v[i + win]
        if drop > best_drop:
            best_drop = drop
            best_i = i
    if best_i is not None and best_drop >= min_drop_kmh:
        return int(best_i)
    return None


# =========================
# 9) Ground landing
# =========================

def ground_landing(df: pd.DataFrame,
                   agl: np.ndarray,
                   alt_tol_m: float = 10.0,
                   speed_tol_mps: float = 1.5,
                   sustain_s: float = 3.0) -> Optional[int]:
    """
    Detects landing: AGL ~ 0 ± alt_tol_m and low ground speed, sustained for sustain_s.
    """
    v = _ground_speed_mps(df)
    t = _to_datetime_utc(df["time"])
    f, dt_mode = determine_frequency(df)
    if not np.isfinite(dt_mode) or dt_mode <= 0:
        dt_mode = 1.0
    win = max(1, int(round(sustain_s / dt_mode)))
    cond = (np.abs(agl) <= alt_tol_m) & (v <= speed_tol_mps)
    sus = _rolling_sustain(cond, win)
    idx = np.argmax(sus) if sus.any() else None
    if idx is None or sus[idx] == 0:
        return None
    return int(idx)


# =========================
# 10) Max lateral deviation vs theoretical trajectory
# =========================

@dataclass
class LateralDeviationResult:
    max_lateral_m: float
    idx_at_max: int

def max_lateral_deviation(df: pd.DataFrame,
                          idx_ref: int,
                          dest_lat: float,
                          dest_lon: float) -> Optional[LateralDeviationResult]:
    """
    Maximum lateral deviation (m) from the axis [ref_point(drop+9s) -> destination].
    """
    if idx_ref is None:
        return None
    lat0 = float(df.loc[idx_ref, "lat"])
    lon0 = float(df.loc[idx_ref, "lon"])
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    x, y = _latlon_to_local_xy(lat, lon, lat0, lon0)
    vx, vy = _latlon_to_local_xy(np.array([dest_lat]), np.array([dest_lon]), lat0, lon0)
    ax, ay = float(vx[0]), float(vy[0])
    v = np.array([ax, ay])
    vn = v / (np.linalg.norm(v) + 1e-9)

    # Signed distance: det(v, r) / ||v||
    r = np.vstack([x, y]).T
    det = v[0]*r[:,1] - v[1]*r[:,0]
    lateral = det / (np.linalg.norm(v) + 1e-9)
    j = int(np.nanargmax(np.abs(lateral)))
    return LateralDeviationResult(max_lateral_m=float(lateral[j]), idx_at_max=j)


# =========================
# 11) Average distance & speed during glide phase
# =========================

def glide_distance_2500_1500(df: pd.DataFrame, agl: np.ndarray,
                             idx_glide_start: int, idx_glide_end: int) -> float:
    """
    Cumulative horizontal distance (m) between idx_glide_start and idx_glide_end (inclusive).
    """
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return float("nan")
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    d = 0.0
    for i in range(idx_glide_start+1, idx_glide_end+1):
        d += _haversine_m(lat[i-1], lon[i-1], lat[i], lon[i])
    return d

def average_horizontal_speed_glide(df: pd.DataFrame, idx_glide_start: int, idx_glide_end: int) -> float:
    """
    Average horizontal speed (m/s) during the glide (arithmetic mean of ground speed).
    """
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return float("nan")
    v = _ground_speed_mps(df)
    return float(np.nanmean(v[idx_glide_start:idx_glide_end+1]))

def glide_time_2500_1500(df: pd.DataFrame, idx_glide_start: int, idx_glide_end: int) -> float:
    """
    Time spent (s) between first passage <2500 m and passage <1500 m.
    """
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return float("nan")
    t = _to_datetime_utc(df["time"])
    return float((t.iloc[idx_glide_end] - t.iloc[idx_glide_start]).total_seconds())


# =========================
# 12) Corrections (geometry / time / speed)
# =========================

def correct_lateral_deviation(df: pd.DataFrame,
                               idx_start: int,
                               idx_end: int,
                               idx_ref: int,
                               dest_lat: float,
                               dest_lon: float,
                               max_deviation_m: float = 50.0,
                               max_shift_per_point_m: float = 10.0,
                               noise_scale: float = 1.0) -> pd.DataFrame:
    """
    Adjusts the trajectory between idx_start..idx_end to |lateral deviation| <= max_deviation_m
    by partial projection towards the axis [ref->dest], limiting per-point displacement
    and reintroducing realistic noise on the lateral component.
    Returns a COPY of the DataFrame with columns lat_corr/lon_corr (originals remain unchanged).
    """
    out = df.copy()
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    lat0 = float(df.loc[idx_ref, "lat"])
    lon0 = float(df.loc[idx_ref, "lon"])

    x, y = _latlon_to_local_xy(lat, lon, lat0, lon0)
    vx, vy = _latlon_to_local_xy(np.array([dest_lat]), np.array([dest_lon]), lat0, lon0)
    ax, ay = float(vx[0]), float(vy[0])
    v = np.array([ax, ay])
    vnorm = np.linalg.norm(v) + 1e-9
    vn = v / vnorm

    r = np.vstack([x, y]).T
    det = v[0]*r[:,1] - v[1]*r[:,0]
    lateral = det / vnorm  # signed lateral distance
    # Unit lateral component (perpendicular to vn, +90° rotation)
    n_hat = np.array([-vn[1], vn[0]])

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)

    # Act on the segment [idx_start:idx_end]
    for i in range(idx_start, idx_end+1):
        lat_err = lateral[i]
        excess = abs(lat_err) - max_deviation_m
        if excess > 0:
            # Shift towards the axis, limited by max_shift_per_point_m
            shift = np.clip(excess, 0, max_shift_per_point_m)
            # Opposite sign to the deviation
            sgn = -np.sign(lat_err)
            dxy = sgn * shift * n_hat
            dx[i] += dxy[0]
            dy[i] += dxy[1]

    # Smooth displacements to avoid sharp changes
    if (idx_end - idx_start) >= 5:
        kernel = 7
        fil = pd.Series(dx).rolling(kernel, center=True, min_periods=1).mean().to_numpy()
        fjl = pd.Series(dy).rolling(kernel, center=True, min_periods=1).mean().to_numpy()
        dx, dy = fil, fjl

    # Realistic lateral noise
    # Estimate the original lateral residual (vs moving average) and reintroduce it
    lat_res = pd.Series(lateral).diff().fillna(0).to_numpy()
    noise = _noise_like(lat_res, scale=noise_scale)
    dx += noise * n_hat[0]
    dy += noise * n_hat[1]

    x_corr = x + dx
    y_corr = y + dy
    lat_corr, lon_corr = _local_xy_to_latlon(x_corr, y_corr, lat0, lon0)
    out["lat_corr"] = lat_corr
    out["lon_corr"] = lon_corr
    return out


def correct_glide_time(df: pd.DataFrame,
                       idx_glide_start: int,
                       idx_glide_end: int,
                       time_factor: float = 1.10) -> pd.DataFrame:
    """
    Dilates the time of the glide segment by time_factor (>1 => longer, <1 => shorter),
    and shifts all subsequent timestamps by a consistent value.
    Local sampling frequency is preserved (same steps within the segment).
    """
    out = df.copy()
    t = _to_datetime_utc(out["time"])
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return out
    t0 = t.iloc[idx_glide_start]
    t1 = t.iloc[idx_glide_end]
    duration = (t1 - t0).total_seconds()
    delta = duration * (time_factor - 1.0)

    # Affine recalibration on the segment
    seg = (t >= t0) & (t <= t1)
    alpha = (t[seg] - t0).dt.total_seconds().to_numpy() / max(duration, 1e-6)
    t_seg_new = t0 + pd.to_timedelta(alpha * duration * time_factor, unit="s")

    # Update
    out.loc[seg, "time"] = t_seg_new.values

    # Shift subsequent points
    after = t > t1
    out.loc[after, "time"] = (t[after] + pd.to_timedelta(delta, unit="s")).values
    return out


def correct_glide_distance_with_points(df: pd.DataFrame,
                                       idx_glide_start: int,
                                       idx_glide_end: int,
                                       idx_ref: int,
                                       dest_lat: float,
                                       dest_lon: float,
                                       distance_factor: float = 1.10,
                                       nominal_freq_hz: Optional[float] = None,
                                       agl: Optional[np.ndarray] = None,
                                       blend_recalibration_m: Tuple[float, float] = (600.0, 300.0),
                                       noise_scale: float = 1.0) -> pd.DataFrame:
    """
    Stretches the distance traveled during the glide by distance_factor (>1 => longer),
    by moving points along the axis [ref->dest], then resamples at the nominal frequency
    to maintain consistent sampling. Between ~blend_recalibration_m (start, end),
    progressively blends back to the original trajectory to match the real trace
    before opening/landing (ground return point remains unchanged).
    """
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return df.copy()
    out = df.copy()
    if nominal_freq_hz is None:
        nominal_freq_hz, _dt = determine_frequency(df)
        if not np.isfinite(nominal_freq_hz):
            nominal_freq_hz = 5.0
    dt_nom = 1.0 / nominal_freq_hz

    # Local reference
    lat0 = float(df.loc[idx_ref, "lat"])
    lon0 = float(df.loc[idx_ref, "lon"])
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    x, y = _latlon_to_local_xy(lat, lon, lat0, lon0)
    vx, vy = _latlon_to_local_xy(np.array([dest_lat]), np.array([dest_lon]), lat0, lon0)
    v = np.array([float(vx[0]), float(vy[0])])
    vn = v / (np.linalg.norm(v) + 1e-9)

    # Extract glide segment
    xs = x[idx_glide_start:idx_glide_end+1].copy()
    ys = y[idx_glide_start:idx_glide_end+1].copy()

    # Component along the axis
    r = np.vstack([xs, ys]).T
    r0 = np.array([x[idx_glide_start], y[idx_glide_start]])
    s = (r - r0) @ vn  # Projected curvilinear abscissa
    s_scaled = (s - s.min()) * distance_factor + s.min()

    # Reconstruct corrected positions (along the axis + original lateral)
    # Original lateral (perpendicular)
    n_hat = np.array([-vn[1], vn[0]])
    lateral_comp = (r - r0) @ n_hat
    r_corr = r0 + np.outer(s_scaled, vn) + np.outer(lateral_comp, n_hat)

    # Realistic noise (on increments)
    inc_lat = np.diff(lateral_comp, prepend=lateral_comp[0])
    noise = _noise_like(inc_lat, scale=noise_scale)
    r_corr += np.outer(noise, n_hat)

    # Resample at dt_nom
    t = _to_datetime_utc(df["time"])
    t_seg = t.iloc[idx_glide_start:idx_glide_end+1]
    t0, t1 = t_seg.iloc[0], t_seg.iloc[-1]
    n_new = max(2, int(round((t1 - t0).total_seconds() / dt_nom)) + 1)
    t_new = pd.date_range(t0, t1, periods=n_new)

    # Linear interpolation
    # Note: interpolate on time (nanoseconds)
    ts = t_seg.view("int64").to_numpy()
    tx = t_new.view("int64").to_numpy()
    x_new = np.interp(tx, ts, r_corr[:,0])
    y_new = np.interp(tx, ts, r_corr[:,1])

    # Blend recalibration (if AGL provided)
    if agl is not None and np.isfinite(agl).any():
        agl_seg = agl[idx_glide_start:idx_glide_end+1]
        agl_new = np.interp(tx, ts, agl_seg)
        a0, a1 = blend_recalibration_m
        # Weight 1 -> corrected, 0 -> original, decreasing as we approach a1
        w = np.clip((agl_new - a1) / max(a0 - a1, 1e-6), 0, 1)
        x_orig = np.interp(tx, ts, x[idx_glide_start:idx_glide_end+1])
        y_orig = np.interp(tx, ts, y[idx_glide_start:idx_glide_end+1])
        x_new = w * x_new + (1 - w) * x_orig
        y_new = w * y_new + (1 - w) * y_orig

    # Assemble DataFrame: before glide (unchanged), segment (replaced), after (unchanged)
    lat_new_seg, lon_new_seg = _local_xy_to_latlon(x_new, y_new, lat0, lon0)

    out_seg = out.iloc[idx_glide_start:idx_glide_end+1].copy()
    out_seg["time"] = t_new
    out_seg["lat_corr"] = lat_new_seg
    out_seg["lon_corr"] = lon_new_seg

    out_before = out.iloc[:idx_glide_start].copy()
    out_after  = out.iloc[idx_glide_end+1:].copy()

    # Concatenate and reindex
    out2 = pd.concat([out_before, out_seg, out_after], ignore_index=True)
    return out2


def correct_ground_speed_glide(df: pd.DataFrame,
                               idx_glide_start: int,
                               idx_glide_end: int,
                               idx_ref: int,
                               dest_lat: float,
                               dest_lon: float,
                               speed_factor: float = 1.05,
                               noise_scale: float = 1.0) -> pd.DataFrame:
    """
    Modifies the geometry of the glide phase to apply a factor on ground speed,
    by moving points along the axis [ref->dest] while preserving the original timestamps.
    """
    if idx_glide_start is None or idx_glide_end is None or idx_glide_end <= idx_glide_start:
        return df.copy()
    out = df.copy()
    t = _to_datetime_utc(df["time"]).to_numpy()

    lat0 = float(df.loc[idx_ref, "lat"])
    lon0 = float(df.loc[idx_ref, "lon"])
    lat = df["lat"].astype(float).to_numpy()
    lon = df["lon"].astype(float).to_numpy()
    x, y = _latlon_to_local_xy(lat, lon, lat0, lon0)
    vx, vy = _latlon_to_local_xy(np.array([dest_lat]), np.array([dest_lon]), lat0, lon0)
    v = np.array([float(vx[0]), float(vy[0])])
    vn = v / (np.linalg.norm(v) + 1e-9)
    n_hat = np.array([-vn[1], vn[0]])

    sl = slice(idx_glide_start, idx_glide_end+1)
    xs = x[sl].copy()
    ys = y[sl].copy()

    # Decomposition (longitudinal + lateral)
    r = np.vstack([xs, ys]).T
    r0 = r[0]
    s = (r - r0) @ vn       # Along the axis
    l = (r - r0) @ n_hat    # Lateral (preserved)

    # Apply factor on longitudinal increments (speed ~ ds/dt)
    ds = np.diff(s, prepend=s[0])
    ds_new = ds * speed_factor
    s_new = np.cumsum(ds_new)
    r_new = r0 + np.outer(s_new, vn) + np.outer(l, n_hat)

    # Light noise
    noise = _noise_like(np.diff(l, prepend=l[0]), scale=noise_scale)
    r_new += np.outer(noise, n_hat)

    x_new, y_new = r_new[:,0], r_new[:,1]
    lat_new, lon_new = _local_xy_to_latlon(x_new, y_new, lat0, lon0)
    out.loc[sl, "lat_corr"] = lat_new
    out.loc[sl, "lon_corr"] = lon_new
    return out

# =========================
# 13) Visualization of time/altitude metrics
# =========================


def relocateDataset(
    datasets: List[Tuple[str, pd.DataFrame]] | List[pd.DataFrame],
    target_agl: float,
) -> List[Tuple[str, pd.DataFrame]] | List[pd.DataFrame]:
    """
    Synchronize multiple datasets on the first dataset by aligning the index
    (>= start_index) where a given AGL (target_agl) is reached.

    Rules:
        - First dataset is the reference (unchanged).
        - Synchronization point = index in reference (>= start_index) whose AGL is
        closest to target_agl.
        - For every other dataset:
            * Find its own closest index >= start_index to target_agl.
            * If its index < ref_sync_index: pad the front with copies of the first row
            (duplicate timestamps as-is) so its sync index shifts right.
            * If its index > ref_sync_index: drop the first rows so that the sync index shifts left.

    Returns a new list with adjusted DataFrames (labels preserved if provided).
    """
    if not datasets:
        return datasets

    # Normalize input shape (with labels or not)
    with_labels = isinstance(datasets[0], tuple) and len(datasets[0]) == 2
    if with_labels:
        labels, dfs = zip(*datasets)
        dfs = list(dfs)
    else:
        dfs = list(datasets)

    def compute_agl(df: pd.DataFrame) -> np.ndarray:
        idx_ascent = start_aircraft_ascent(df)
        g = average_ground_altitude(df, idx_ascent)
        return agl_series(df, g)

    # Reference dataset
    ref_df = dfs[0].copy()
    ref_agl = compute_agl(ref_df)
    drop_index=start_drop(ref_df,ref_agl)
    ref_start_index = reference_point(ref_df,drop_index)
    if ref_start_index >= len(ref_agl):
        return datasets  # nothing to do

    # Find reference sync index: index >= start_index minimizing |AGL - target|
    ref_slice = np.arange(ref_start_index, len(ref_agl))
    ref_idx = int(ref_slice[np.argmin(np.abs(ref_agl[ref_slice] - target_agl))])

    out_dfs: List[pd.DataFrame] = [ref_df]

    for k in range(1, len(dfs)):
        df = dfs[k].copy()
        agl = compute_agl(df)
        drop_index=start_drop(df,agl)
        start_index=reference_point(df,drop_index)
        if start_index >= len(agl):
            out_dfs.append(df)
            continue

        slice_idx = np.arange(start_index, len(agl))
        sync_idx = int(slice_idx[np.argmin(np.abs(agl[slice_idx] - target_agl))])

        if sync_idx == ref_idx:
            out_dfs.append(df)
            continue

        if sync_idx < ref_idx:
            # Need to pad (ref_idx - sync_idx) rows at the front
            pad_n = ref_idx - sync_idx
            first_row = df.iloc[0:1].copy()
            pads = pd.concat([first_row] * pad_n, ignore_index=True)
            df = pd.concat([pads, df], ignore_index=True)
        else:
            # Need to drop first (sync_idx - ref_idx) rows
            drop_n = sync_idx - ref_idx
            if drop_n < len(df):
                df = df.iloc[drop_n:].reset_index(drop=True)
            else:
                # Degenerate: all dropped, keep a single repeated row
                df = pd.DataFrame([dfs[k].iloc[0].to_dict()])

        out_dfs.append(df)

    if with_labels:
        return list(zip(labels, out_dfs))
    return out_dfs



def plot_time_altitude_metrics(
    datasets: List[Tuple[str, pd.DataFrame]],
    vertical_sign: str = "down",
    shade_alpha: float = 0.12,
    glidezone_startagl: float = 2500.0,
    glidezone_stopagl: float = 1500.0,
) -> None:
    """
    Single figure (no subplots) with multiple y-axes:
      - Altitude (left)
      - Horizontal speed, Vertical speed, Glide ratio (stacked on right with offset spines)
    Each metric has its own axis; time aligned on reference dataset at glidezone_startagl.
    """
    if not datasets:
        return

    labels, dfs = zip(*datasets)

    def prep(df: pd.DataFrame):
        idx_ascent = start_aircraft_ascent(df)
        ground = average_ground_altitude(df, idx_ascent)
        agl = agl_series(df, ground)
        t = _to_datetime_utc(df["time"])
        vh = _ground_speed_mps(df)
        vz_up = _vertical_speed_up_mps(df)
        ref = reference_point(df, idx_drop=start_drop(df, agl))
        low = agl_index(df, agl, idx_start_search=ref, threshold=1000)
        pd = parachute_deployment(df, agl, startidx=ref)
        stglide= start_glide_2500(df, agl, idx_start_search=ref)
        endglide= end_glide_1500(df, agl, idx_glide_start=stglide)
        vz = -vz_up if vertical_sign.lower() == "down" else vz_up
        if vertical_sign.lower() == "down":
            denom = vz.copy()
        else:
            denom = np.maximum(-vz_up, 0.0)
        gr = np.full_like(vh, np.nan)
        mask = denom > 0.3
        gr[mask] = vh[mask] / denom[mask]
        return t, agl, vh, vz, gr, ref, low, pd, stglide, endglide

    ref_t, ref_agl, *_ = prep(dfs[0])
    ref_idx_candidates = np.where(np.isfinite(ref_agl))[0]
    if ref_idx_candidates.size == 0:
        return
    ref_idx = int(ref_idx_candidates[np.argmin(np.abs(ref_agl[ref_idx_candidates] - glidezone_startagl))])
    t0_ref = ref_t.iloc[ref_idx]

    import matplotlib.colors as mcolors
    metric_colors = {
        "alt": "#1f77b4",
        "vh": "#ff7f0e",
        "vz": "#2ca02c",
        "gr": "#d62728",
    }

    def variant(color, i, n):
        rgb = np.array(mcolors.to_rgb(color))
        if n <= 1:
            return rgb
        w = 0.6 * (1 - i / max(n - 1, 1))
        return rgb * (1 - w) + w * 1.0

    fig, ax_alt = plt.subplots(figsize=(11, 6))
    ax_alt.set_ylabel("AGL (m)", color=metric_colors["alt"])
    ax_alt.tick_params(axis='y', colors=metric_colors["alt"])
    ax_alt.grid(True, alpha=0.35)

    # Create additional axes
    ax_vh = ax_alt.twinx()
    ax_vh.set_ylabel("Horiz V (m/s)", color=metric_colors["vh"])
    ax_vh.tick_params(axis='y', colors=metric_colors["vh"])

    ax_vz = ax_alt.twinx()
    ax_vz.set_ylabel(f"Vert V ({'down' if vertical_sign.lower()=='down' else 'up'}) (m/s)", color=metric_colors["vz"])
    ax_vz.tick_params(axis='y', colors=metric_colors["vz"])

    ax_gr = ax_alt.twinx()
    ax_gr.set_ylabel("Glide ratio", color=metric_colors["gr"])
    ax_gr.tick_params(axis='y', colors=metric_colors["gr"])

    # Offset right-side spines
    axes_right = [ax_vh, ax_vz, ax_gr]
    offsets = [0.0, 60, 120]  # pts
    for ax, off in zip(axes_right, offsets):
        ax.spines["right"].set_position(("outward", off))
        ax.spines["right"].set_visible(True)

    # Shaded altitude band
    ax_alt.axhspan(glidezone_stopagl, glidezone_startagl, color="gray", alpha=shade_alpha, zorder=0)

    lines_for_legend = []
    metric_legend_handles = [
        plt.Line2D([], [], color=metric_colors["alt"], label="Altitude"),
        plt.Line2D([], [], color=metric_colors["vh"], label="Horizontal Speed"),
        plt.Line2D([], [], color=metric_colors["vz"], label="Vertical Speed"),
        plt.Line2D([], [], color=metric_colors["gr"], label="Glide Ratio"),
    ]

    for i, (label, df) in enumerate(zip(labels, dfs)):
        t, agl, vh, vz, gr, ref, low,pd, stglide, endglide = prep(df)

        # Time axis (seconds from reference alignment point of reference dataset)
        t0=t.iloc[stglide]
        t=t[ref:low]
        time_s = (t - t0).dt.total_seconds().to_numpy()

        c_alt = variant(metric_colors["alt"], i, len(dfs))
        c_vh  = variant(metric_colors["vh"],  i, len(dfs))
        c_vz  = variant(metric_colors["vz"],  i, len(dfs))
        c_gr  = variant(metric_colors["gr"],  i, len(dfs))

        agl= agl[ref:low]
        l_alt, = ax_alt.plot(time_s[:len(agl)], agl, color=c_alt, lw=1.3,
                             label=label)  # dataset legend from altitude lines
        vh=vh[ref:low]
        vz=vz[ref:low]
        gr=gr[ref:low]

        ax_vh.plot(time_s[:len(vh)], vh, color=c_vh, lw=1.0)
        ax_vz.plot(time_s[:len(vz)], vz, color=c_vz, lw=1.0)
        ax_gr.plot(time_s[:len(gr)], gr, color=c_gr, lw=1.0)

        lines_for_legend.append(l_alt)

    ax_alt.set_xlabel("Time aligned (s)")

    # Dataset legend (top-left)
    if lines_for_legend:
        ax_alt.legend(handles=lines_for_legend, title="Datasets", loc="upper left")

    # Metric legend (top-right)
    ax_alt.legend(handles=metric_legend_handles, title="Metrics", loc="upper right")
    # Combine both legends (keep order)
    handles = lines_for_legend + metric_legend_handles
    labels_all = [h.get_label() for h in handles]
    ax_alt.legend(handles, labels_all, loc="lower right", framealpha=0.85)

    plt.tight_layout()
    plt.show()
