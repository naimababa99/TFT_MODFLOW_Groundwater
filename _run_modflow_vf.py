import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import flopy
import joblib
import time

warnings.filterwarnings('ignore')

# =============================================================================
# 1. ENHANCED CONFIGURATION
# =============================================================================
class EnhancedMODFLOWConfig:
    """Enhanced configuration with time-varying parameters."""
    
    def __init__(
        self,
        model_name: str = 'gwl_model',
        model_ws: str = 'modflow_workspace',
        exe_name: str = 'mf2005',
        # Grid parameters
        nlay: int = 2,  # Multiple layers for better vertical flow
        nrow:  int = 15,
        ncol: int = 15,
        delr: float = 500.0,  # Finer grid
        delc: float = 500.0,
        # Layer properties
        top: float = 100.0,
        botm: List[float] = None,
        # Aquifer properties (will be calibrated)
        hk: float = 10.0,
        vka: float = 0.1,  # Vertical anisotropy
        sy: float = 0.15,
        ss: float = 1e-5,
        laytyp: List[int] = None,
        # Initial/boundary conditions
        strt: float = 88.0,
        # Observation well location
        well_row: int = 7,
        well_col: int = 7,
        # Reference elevation
        reference_elevation: float = 100.0,
    ):
        self.model_name = model_name
        self.model_ws = model_ws
        self.exe_name = exe_name
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self. delr = delr
        self.delc = delc
        self.top = top
        self. botm = botm if botm else [50.0, 0.0]  # Two layers
        self.hk = hk
        self.vka = vka
        self.sy = sy
        self.ss = ss
        self. laytyp = laytyp if laytyp else [1, 0]  # Unconfined, confined
        self.strt = strt
        self.well_row = well_row
        self.well_col = well_col
        self.reference_elevation = reference_elevation

# =============================================================================
# 2. DATA PREPARATION (SAME AS BEFORE)
# =============================================================================
class DataPreparator:
    def __init__(
        self, data_path: str, date_col: str = 'date',
        target_col: str = 'water_level',
        train_end:  str = '2021-12-31', val_end: str = '2023-12-31',
    ):
        self.data_path = data_path
        self.date_col = date_col
        self.target_col = target_col
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        print("="*70)
        print("     LOADING DATA")
        print("="*70)
        
        self.df = pd. read_csv(self.data_path, parse_dates=[self. date_col])
        self.df = self. df.sort_values(self.date_col).reset_index(drop=True)
        
        print(f"\n  Dataset:  {len(self. df)} records")
        print(f"  Date range: {self. df[self.date_col].min()} to {self.df[self.date_col].max()}")
        
        return self.df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.target_col] = df[self.target_col].interpolate(method='linear', limit_direction='both')
        df[self.target_col] = df[self.target_col].fillna(method='ffill').fillna(method='bfill')
        
        numeric_cols = df. select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_col:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(0)
        
        return df
    
    def convert_depth_to_head(self, depth_feet: np.ndarray, ref_elev: float = 100.0) -> np.ndarray:
        return ref_elev - np.array(depth_feet, dtype=float) * 0.3048
    
    def convert_head_to_depth(self, head_meters: np.ndarray, ref_elev: float = 100.0) -> np.ndarray:
        return (ref_elev - np.array(head_meters, dtype=float)) / 0.3048

# =============================================================================
# 3. ENHANCED MODFLOW DATA PREPARATION
# =============================================================================
class EnhancedMODFLOWDataPreparator:
    """Prepare data with time-varying recharge and trend analysis."""
    
    def __init__(self, config: EnhancedMODFLOWConfig):
        self.config = config
        
    def calculate_recharge(self, df: pd.DataFrame,
                          precip_col: str = 'total_precipitation_sum',
                          evap_col: str = 'evaporation_sum',
                          temp_col: str = '2m_temperature_mean',
                          recharge_fraction: float = 0.15) -> np.ndarray:
        """Calculate recharge with temperature-dependent ET adjustment."""
        print("\n  Calculating enhanced recharge...")
        
        # Get precipitation
        if precip_col in df.columns:
            precip = np.nan_to_num(df[precip_col].values. copy(), nan=0.0)
            if np.max(np.abs(precip)) < 1:
                precip = precip * 1000
        else:
            precip = np.zeros(len(df))
        
        # Get evaporation
        if evap_col in df.columns:
            evap = np.abs(np.nan_to_num(df[evap_col].values.copy(), nan=0.0))
            if np.max(evap) < 1:
                evap = evap * 1000
        else:
            evap = np. zeros(len(df))
        
        # Temperature-adjusted ET factor
        if temp_col in df.columns:
            temp = df[temp_col]. values
            # Higher temperature = more ET
            temp_factor = 1.0 + 0.02 * (temp - np.mean(temp))
            temp_factor = np.clip(temp_factor, 0.5, 1.5)
        else:
            temp_factor = np. ones(len(df))
        
        # Calculate net recharge
        precip = np.maximum(precip, 0)
        adjusted_evap = evap * temp_factor
        net_water = precip - adjusted_evap
        
        # Apply recharge fraction only to positive values
        recharge = np.where(net_water > 0, net_water * recharge_fraction, 0) / 1000.0
        recharge = np.clip(recharge, 1e-7, 0.005)
        
        print(f"    Recharge range: {recharge.min():.7f} - {recharge.max():.6f} m/day")
        print(f"    Mean recharge: {recharge.mean():.6f} m/day")
        
        return recharge
    
    def analyze_trend(self, observed_heads: np.ndarray, dates: np.ndarray) -> Dict: 
        """Analyze trend in observed data."""
        print("\n  Analyzing observed water level trend...")
        
        # Convert dates to numeric
        if isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            x = (pd.to_datetime(dates) - pd.to_datetime(dates[0])).days.values
        else: 
            x = np.arange(len(observed_heads))
        
        # Fit linear trend
        mask = ~np.isnan(observed_heads)
        if np.sum(mask) > 2:
            coeffs = np.polyfit(x[mask], observed_heads[mask], 1)
            trend_slope = coeffs[0]  # meters per day
            trend_intercept = coeffs[1]
        else:
            trend_slope = 0
            trend_intercept = np.nanmean(observed_heads)
        
        # Estimate required pumping to match trend
        # Head decline = pumping / (area * Sy)
        area = self.config.nrow * self.config. ncol * self.config.delr * self.config.delc
        head_change_per_year = trend_slope * 365  # meters/year
        
        print(f"    Trend slope: {trend_slope*365:.4f} m/year ({trend_slope*365/0.3048:.4f} ft/year)")
        print(f"    Initial head:  {trend_intercept:.2f} m")
        
        return {
            'slope': trend_slope,
            'intercept': trend_intercept,
            'head_change_per_year':  head_change_per_year,
        }
    
    def prepare_monthly_stress_periods(self, df:  pd.DataFrame, recharge:  np.ndarray,
                                       date_col: str = 'date'):
        """Prepare monthly stress periods."""
        print("\n  Preparing monthly stress periods...")
        
        df = df.copy()
        df['recharge'] = recharge
        df['period'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        
        grouped = df.groupby('period').agg({
            'recharge': 'mean',
            date_col: 'first',
        }).reset_index()
        
        perlen = [float(row['period'].days_in_month) for _, row in grouped.iterrows()]
        nstp = [max(1, int(p // 5)) for p in perlen]  # More time steps
        steady = [True] + [False] * (len(perlen) - 1)
        
        recharge_by_period = [max(r if not np.isnan(r) else 1e-6, 1e-7) 
                              for r in grouped['recharge']. tolist()]
        period_dates = grouped[date_col].values
        
        print(f"    Stress periods: {len(perlen)}")
        print(f"    Average time steps per period: {np.mean(nstp):.1f}")
        
        return perlen, nstp, steady, recharge_by_period, period_dates
    
    def aggregate_observed_monthly(self, df:  pd.DataFrame, target_col: str,
                                   date_col: str = 'date'):
        df = df.copy()
        df['period'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        
        grouped = df. groupby('period').agg({
            target_col: 'mean',
            date_col: 'first',
        }).reset_index()
        
        values = np.array(grouped[target_col].values, dtype=float)
        dates = grouped[date_col]. values
        
        if np.any(np. isnan(values)):
            mask = ~np.isnan(values)
            if np.any(mask):
                values = np.interp(np.arange(len(values)), 
                                  np.arange(len(values))[mask], values[mask])
        
        return values, dates

# =============================================================================
# 4. ENHANCED MODFLOW MODEL WITH PUMPING
# =============================================================================
class EnhancedModflowModel: 
    """MODFLOW model with pumping well to capture declining trends."""
    
    def __init__(self, config: EnhancedMODFLOWConfig):
        self.config = config
        
    def build_and_run(
        self,
        perlen: List[float],
        nstp: List[int],
        steady: List[bool],
        recharge_by_period: List[float],
        hk: float,
        sy: float,
        strt: float,
        pumping_rate: float = 0.0,  # NEW:  pumping rate (m³/day, negative for extraction)
        trend_adjustment: float = 0.0,  # NEW: gradual change in boundary heads
        silent: bool = True,
    ) -> Optional[np.ndarray]: 
        
        cfg = self.config
        nper = len(perlen)
        
        os.makedirs(cfg.model_ws, exist_ok=True)
        
        mf = flopy.modflow.Modflow(
            modelname=cfg.model_name,
            exe_name=cfg. exe_name,
            model_ws=cfg. model_ws,
        )
        
        # DIS - Multiple layers
        flopy.modflow.ModflowDis(
            mf, nlay=cfg. nlay, nrow=cfg.nrow, ncol=cfg.ncol,
            delr=cfg.delr, delc=cfg. delc, top=cfg.top, botm=cfg. botm,
            nper=nper, perlen=perlen, nstp=nstp, steady=steady,
            itmuni=4, lenuni=2,
        )
        
        # BAS - Time-varying boundary heads to capture regional trend
        ibound = np.ones((cfg.nlay, cfg.nrow, cfg.ncol), dtype=np.int32)
        ibound[: , :, 0] = -1   # Left boundary - constant head
        ibound[: , :, -1] = -1  # Right boundary - constant head
        
        strt_array = np.ones((cfg.nlay, cfg.nrow, cfg.ncol)) * strt
        flopy.modflow. ModflowBas(mf, ibound=ibound, strt=strt_array)
        
        # LPF - Layer properties
        hk_array = np.ones((cfg. nlay, cfg. nrow, cfg. ncol)) * hk
        sy_array = np. ones((cfg.nlay, cfg.nrow, cfg.ncol)) * sy
        
        flopy.modflow.ModflowLpf(
            mf, laytyp=cfg.laytyp, hk=hk_array, vka=cfg.vka,
            sy=sy_array, ss=cfg.ss, ipakcb=53,
        )
        
        # RCH - Recharge
        rech = {kper: recharge_by_period[kper] for kper in range(nper)}
        flopy.modflow. ModflowRch(mf, rech=rech, ipakcb=53)
        
        # WEL - Pumping well (if specified)
        if pumping_rate != 0:
            # Add pumping well at center of model
            pump_row = cfg.nrow // 2
            pump_col = cfg.ncol // 2
            
            well_spd = {}
            for kper in range(nper):
                # Can make pumping time-varying
                well_spd[kper] = [[0, pump_row, pump_col, pumping_rate]]
            
            flopy.modflow.ModflowWel(mf, stress_period_data=well_spd, ipakcb=53)
        
        # CHD - Time-varying constant head boundaries to simulate regional trend
        if trend_adjustment != 0:
            chd_spd = {}
            cumulative_time = 0
            for kper in range(nper):
                # Calculate head decline for this period
                head_at_boundary = strt + trend_adjustment * cumulative_time
                
                chd_data = []
                for row in range(cfg. nrow):
                    for lay in range(cfg. nlay):
                        # Left boundary
                        chd_data.append([lay, row, 0, head_at_boundary, head_at_boundary])
                        # Right boundary
                        chd_data.append([lay, row, cfg.ncol-1, head_at_boundary, head_at_boundary])
                
                chd_spd[kper] = chd_data
                cumulative_time += perlen[kper]
            
            flopy.modflow. ModflowChd(mf, stress_period_data=chd_spd)
        
        # OC - Output control
        spd = {}
        for kper in range(nper):
            for kstp in range(nstp[kper]):
                spd[(kper, kstp)] = ['save head']
        flopy.modflow. ModflowOc(mf, stress_period_data=spd, compact=True)
        
        # PCG - Solver
        flopy.modflow. ModflowPcg(mf, hclose=1e-4, rclose=1e-4, mxiter=500, iter1=200)
        
        # Write and run
        mf.write_input()
        success, buff = mf.run_model(silent=silent)
        
        if not success: 
            return None
        
        time. sleep(0.1)
        
        # Read heads
        headfile = None
        for fname in [
            os.path. join(cfg.model_ws, f"{cfg.model_name}. hds"),
            os.path.join(cfg.model_ws, f"{cfg.model_name. upper()}.hds"),
        ]: 
            if os.path.exists(fname):
                headfile = fname
                break
        
        if headfile is None: 
            try:
                files = os.listdir(cfg.model_ws)
                hds_files = [f for f in files if f.lower().endswith('.hds')]
                if hds_files: 
                    headfile = os.path.join(cfg.model_ws, hds_files[0])
            except:
                pass
        
        if headfile is None or not os.path. exists(headfile):
            return None
        
        try:
            hds = flopy.utils. HeadFile(headfile)
            times = hds.get_times()
            
            heads_at_well = []
            for kper in range(nper):
                sp_end_time = sum(perlen[: kper+1])
                closest_time = min(times, key=lambda t: abs(t - sp_end_time))
                head = hds.get_data(totim=closest_time)
                # Get head from top layer at observation well
                heads_at_well. append(head[0, cfg.well_row, cfg. well_col])
            
            return np.array(heads_at_well)
            
        except Exception as e:
            return None

# =============================================================================
# 5. ENHANCED CALIBRATOR WITH PUMPING/TREND PARAMETERS
# =============================================================================
class EnhancedCalibrator: 
    """Calibrator with additional parameters for trend capture."""
    
    def __init__(
        self,
        config: EnhancedMODFLOWConfig,
        perlen: List[float],
        nstp:  List[int],
        steady: List[bool],
        recharge_by_period: List[float],
        observed_heads: np. ndarray,
        trend_info: Dict = None,
        train_cutoff_idx: Optional[int] = None,
    ):
        """
        train_cutoff_idx: LEAKAGE FIX. Number of leading monthly stress
        periods that fall within the TRAIN period (date <= train_end).
        When set, the differential-evolution objective in _run_model /
        _objective_function is computed ONLY over observed_heads[:train_cutoff_idx]
        vs. simulated[:train_cutoff_idx] -- i.e. the physical parameters
        (hk, sy, strt, rch_mult, pumping) are calibrated against the train
        period only. The full simulated series (all periods, including
        val/test) is still returned so it can be used downstream as a
        physics prior / for held-out evaluation -- it just never
        influences which parameters are chosen. Previously this was None
        everywhere, so calibration minimized RMSE against observed heads
        across the ENTIRE record (train+val+test combined), meaning the
        physics prior was fit using the same val/test observations the
        Hybrid model is later evaluated against.
        """
        self.config = config
        self.perlen = perlen
        self.nstp = nstp
        self. steady = steady
        self.recharge_by_period = recharge_by_period
        self.observed_heads = observed_heads
        self.trend_info = trend_info or {}
        self.train_cutoff_idx = train_cutoff_idx
        
        self.best_params = None
        self.best_rmse = float('inf')
        self.best_r2 = -999
        self.iteration = 0
        self.results = []
        
        # Extended parameter bounds
        self.bounds = {
            'hk': (1.0, 100.0),
            'sy': (0.05, 0.30),
            'strt': (86.0, 92.0),
            'rch_mult': (0.1, 5.0),
            'pumping':  (-1000.0, 0.0),  # Negative = extraction
        }
        
        # Adjust strt bounds based on observed data (train-only, if we know
        # the cutoff -- otherwise this itself would leak val/test range
        # information into the parameter search space)
        obs_for_bounds = (
            observed_heads[:train_cutoff_idx] if train_cutoff_idx is not None
            else observed_heads
        )
        if len(obs_for_bounds) > 0:
            self.bounds['strt'] = (
                np.nanmin(obs_for_bounds) - 2,
                np.nanmax(obs_for_bounds) + 2
            )
    
    def _run_model(self, hk:  float, sy: float, strt: float, 
                   rch_mult: float, pumping: float) -> Tuple[float, float, Optional[np.ndarray]]:
        """Run a single model with given parameters."""
        
        calib_ws = os.path.join(self.config.model_ws, f'calib_{self.iteration}')
        
        try:
            if os.path.exists(calib_ws):
                shutil.rmtree(calib_ws, ignore_errors=True)
                time.sleep(0.05)
            os.makedirs(calib_ws, exist_ok=True)
            
            calib_config = EnhancedMODFLOWConfig(
                model_name='calib',
                model_ws=calib_ws,
                exe_name=self.config.exe_name,
                nlay=self.config.nlay,
                nrow=self.config.nrow,
                ncol=self.config. ncol,
                delr=self. config.delr,
                delc=self.config.delc,
                top=self.config. top,
                botm=self.config.botm,
                well_row=self. config.well_row,
                well_col=self.config.well_col,
            )
            
            scaled_recharge = [r * rch_mult for r in self.recharge_by_period]
            
            # Calculate trend adjustment from observed data
            if self.trend_info:
                trend_adjustment = self.trend_info.get('slope', 0)
            else: 
                trend_adjustment = 0
            
            model = EnhancedModflowModel(calib_config)
            simulated = model.build_and_run(
                self.perlen, self.nstp, self.steady, scaled_recharge,
                hk=hk, sy=sy, strt=strt,
                pumping_rate=pumping,
                trend_adjustment=trend_adjustment,
                silent=True
            )
            
            if simulated is None: 
                return float('inf'), -999, None
            
            n = min(len(simulated), len(self.observed_heads))
            
            # LEAKAGE FIX: the calibration objective (what differential_evolution
            # actually minimizes) is computed on the TRAIN-period slice only.
            # `simulated` returned below is still the full-length series
            # (all stress periods), so callers can evaluate val/test
            # performance separately without it ever affecting parameter choice.
            if self.train_cutoff_idx is not None:
                obj_n = min(n, self.train_cutoff_idx)
            else:
                obj_n = n
            
            sim_obj = simulated[:obj_n]
            obs_obj = self.observed_heads[:obj_n]
            
            rmse = np.sqrt(np.mean((sim_obj - obs_obj) ** 2))
            
            ss_res = np.sum((sim_obj - obs_obj) ** 2)
            ss_tot = np.sum((obs_obj - np. mean(obs_obj)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
            
            return rmse, r2, simulated
            
        except Exception as e:
            return float('inf'), -999, None
        finally:
            try:
                if os.path.exists(calib_ws):
                    shutil.rmtree(calib_ws, ignore_errors=True)
            except: 
                pass
    
    def _objective_function(self, params: np.ndarray) -> float:
        hk, sy, strt, rch_mult, pumping = params
        
        self.iteration += 1
        rmse, r2, _ = self._run_model(hk, sy, strt, rch_mult, pumping)
        
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_r2 = r2
            self. best_params = {
                'hk': hk, 'sy': sy, 'strt': strt, 
                'rch_mult': rch_mult, 'pumping': pumping
            }
            print(f"    Iter {self.iteration}:  RMSE={rmse:.4f} (train-only), R²={r2:.4f} (NEW BEST)")
        elif self.iteration % 25 == 0:
            print(f"    Iter {self.iteration}:  RMSE={rmse:.4f} (train-only)")
        
        self.results.append({
            'iteration':  self.iteration,
            'hk': hk, 'sy': sy, 'strt': strt, 
            'rch_mult': rch_mult, 'pumping': pumping,
            'rmse': rmse, 'r2':  r2
        })
        
        return rmse if rmse < float('inf') else 1e10
    
    def calibrate(self, maxiter: int = 80, popsize: int = 15, seed: int = 42):
        """Run differential evolution calibration."""
        print("\n" + "="*70)
        print("     ENHANCED CALIBRATION:  DIFFERENTIAL EVOLUTION")
        print("="*70)
        
        if self.train_cutoff_idx is not None:
            print(f"\n  LEAKAGE FIX ACTIVE: calibration objective restricted to the")
            print(f"  first {self.train_cutoff_idx} of {len(self.observed_heads)} monthly periods (train period only).")
        else:
            print(f"\n  WARNING: train_cutoff_idx not set -- calibrating against the")
            print(f"  FULL observed record (train+val+test). This leaks val/test")
            print(f"  observations into the physics prior's calibration.")
        
        self.iteration = 0
        self.results = []
        
        bounds_list = [
            self.bounds['hk'], self.bounds['sy'],
            self.bounds['strt'], self.bounds['rch_mult'],
            self.bounds['pumping'],
        ]
        
        print(f"\n  Parameters:  5 (including pumping)")
        print(f"  Max iterations: {maxiter}, Population:  {popsize}")
        print(f"\n  Parameter bounds:")
        for name, (low, high) in self.bounds.items():
            print(f"    {name}:  {low} - {high}")
        
        print(f"\n  Running optimization...")
        
        result = differential_evolution(
            self._objective_function, bounds_list,
            maxiter=maxiter, popsize=popsize,
            mutation=(0.5, 1.0), recombination=0.7,
            seed=seed, disp=False, workers=1, updating='deferred',
            polish=True,  # Local optimization at the end
        )
        
        self._print_results()
        return self.best_params
    
    def _print_results(self):
        if self.best_params is None:
            print("\n  ✗ No successful runs!")
            return
        
        print(f"\n  " + "="*50)
        print(f"  CALIBRATION RESULTS")
        print(f"  " + "="*50)
        print(f"\n  Best RMSE (train-only): {self.best_rmse:.4f} m")
        print(f"  Best R² (train-only):    {self.best_r2:.4f}")
        print(f"\n  Best Parameters:")
        for name, value in self. best_params.items():
            print(f"    {name}: {value:.4f}")


# =============================================================================
# 6. METRICS AND PLOTTING
# =============================================================================
def calculate_metrics(observed:  np.ndarray, simulated: np. ndarray) -> Dict:
    n = min(len(observed), len(simulated))
    obs, sim = observed[:n], simulated[:n]
    
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    
    if len(obs) == 0:
        return {'RMSE': np. nan, 'MAE': np.nan, 'R2': np.nan, 'NSE': np.nan}
    
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    mae = np.mean(np.abs(sim - obs))
    ss_res = np. sum((obs - sim) ** 2)
    ss_tot = np. sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'NSE': r2}


def plot_enhanced_results(dates, observed, simulated, metrics, save_path=None):
    fig, axes = plt. subplots(2, 2, figsize=(14, 10))
    
    n = min(len(observed), len(simulated), len(dates))
    dates, observed, simulated = dates[:n], observed[:n], simulated[:n]
    
    ax1 = axes[0, 0]
    ax1.plot(dates, observed, 'b-', label='Observed', linewidth=2)
    ax1.plot(dates, simulated, 'r--', label='Simulated', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Depth to Water (feet)')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[0, 1]
    ax2.scatter(observed, simulated, alpha=0.6, s=40)
    lims = [min(observed. min(), simulated.min()), max(observed.max(), simulated.max())]
    ax2.plot(lims, lims, 'k--', linewidth=2)
    ax2.set_xlabel('Observed (feet)')
    ax2.set_ylabel('Simulated (feet)')
    ax2.set_title(f'Scatter Plot (R² = {metrics["R2"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    residuals = simulated - observed
    ax3.plot(dates, residuals, 'g-', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Residual (feet)')
    ax3.set_title(f'Residuals (RMSE = {metrics["RMSE"]:.3f} ft)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Residual (feet)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('MODFLOW Calibration Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved:  {save_path}")
    
    plt.show()

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================
def run_enhanced_modflow_pipeline(
    data_path: str,
    model_dir: str,
    target_col: str = 'water_level',
    date_col: str = 'date',
    train_end:  str = '2021-12-31',
    val_end: str = '2023-12-31',
    exe_name: str = 'mf2005',
    recharge_fraction: float = 0.15,
    reference_elevation: float = 100.0,
    maxiter: int = 80,
    popsize: int = 15,
):
    """Run enhanced MODFLOW calibration pipeline."""
    
    print("\n" + "█"*70)
    print("   ENHANCED MODFLOW CALIBRATION PIPELINE")
    print("   With Pumping & Trend Adjustment")
    print("█"*70)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    data_prep = DataPreparator(
        data_path=data_path, date_col=date_col, target_col=target_col,
        train_end=train_end, val_end=val_end,
    )
    
    df = data_prep. load_data()
    df = data_prep. clean_data(df)
    
    # Unit conversion
    print("\n" + "="*70)
    print("     UNIT CONVERSION & TREND ANALYSIS")
    print("="*70)
    
    observed_depth_ft = np.nan_to_num(df[target_col].values, nan=38.0)
    observed_head_m = data_prep.convert_depth_to_head(observed_depth_ft, reference_elevation)
    
    print(f"\n  Observed depth (feet): {np.min(observed_depth_ft):.2f} - {np. max(observed_depth_ft):.2f}")
    print(f"  Converted to head (m): {np.min(observed_head_m):.2f} - {np.max(observed_head_m):.2f}")
    
    # MODFLOW configuration
    config = EnhancedMODFLOWConfig(
        model_name='gw_enhanced',
        model_ws=model_dir,
        exe_name=exe_name,
        nlay=2,
        nrow=15,
        ncol=15,
        delr=500.0,
        delc=500.0,
        top=reference_elevation,
        botm=[50.0, 0.0],
        well_row=7,
        well_col=7,
        reference_elevation=reference_elevation,
    )
    
    # Prepare data
    modflow_prep = EnhancedMODFLOWDataPreparator(config)
    recharge = modflow_prep. calculate_recharge(df, recharge_fraction=recharge_fraction)
    
    # Analyze trend
    # LEAKAGE FIX: `trend_adjustment` (a fixed, non-calibrated boundary-head
    # drift term derived from this trend) directly shapes the simulated
    # heads in every period, including val/test -- so fitting its
    # slope/intercept on the full record (as before) would let the
    # physics prior "see" the val/test trend even though the DE objective
    # above is already train-only. Restrict the linear fit to the
    # train-period slice (date <= train_end) so this secondary parameter
    # is estimated from train data only, same as the calibrated ones.
    train_end_dt_for_trend = pd.to_datetime(train_end)
    train_mask_daily = np.asarray(pd.to_datetime(df[date_col].values) <= train_end_dt_for_trend)
    temp_df = df.copy()
    temp_df['head_m'] = observed_head_m
    trend_info = modflow_prep.analyze_trend(
        observed_head_m[train_mask_daily],
        df[date_col].values[train_mask_daily],
    )
    
    perlen, nstp, steady, recharge_by_period, period_dates = \
        modflow_prep.prepare_monthly_stress_periods(df, recharge, date_col)
    
    observed_head_monthly, _ = modflow_prep.aggregate_observed_monthly(temp_df, 'head_m', date_col)
    observed_depth_monthly, _ = modflow_prep.aggregate_observed_monthly(df, target_col, date_col)
    
    # LEAKAGE FIX: determine how many of the monthly stress periods fall
    # within the TRAIN period (date <= train_end). This index is passed to
    # EnhancedCalibrator so differential_evolution only ever sees train-period
    # observations when choosing (hk, sy, strt, rch_mult, pumping) -- val/test
    # periods are simulated (using the calibrated parameters) but never
    # influence which parameters are chosen.
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    period_dates_dt = pd.to_datetime(period_dates)
    train_cutoff_idx = int(np.sum(period_dates_dt <= train_end_dt))
    val_cutoff_idx = int(np.sum(period_dates_dt <= val_end_dt))
    
    print(f"\n  Monthly stress periods: {len(period_dates)} total")
    print(f"  Train periods (calibration objective): {train_cutoff_idx} "
          f"({period_dates_dt.min().date()} to {train_end_dt.date()})")
    print(f"  Val periods (held out): {val_cutoff_idx - train_cutoff_idx}")
    print(f"  Test periods (held out): {len(period_dates) - val_cutoff_idx}")
    
    # Test model
    print("\n" + "="*70)
    print("     TESTING MODEL")
    print("="*70)
    
    initial_strt = observed_head_monthly[0]  # Start at first observed value
    print(f"\n  Initial head:  {initial_strt:.2f} m")
    
    test_ws = os.path. join(model_dir, 'test_run')
    if os.path.exists(test_ws):
        shutil. rmtree(test_ws, ignore_errors=True)
        time.sleep(0.1)
    os.makedirs(test_ws, exist_ok=True)
    
    test_config = EnhancedMODFLOWConfig(
        model_name='test_model',
        model_ws=test_ws,
        exe_name=exe_name,
        nlay=2, nrow=15, ncol=15,
        delr=500.0, delc=500.0,
        top=reference_elevation, botm=[50.0, 0.0],
        well_row=7, well_col=7,
    )
    
    test_model = EnhancedModflowModel(test_config)
    test_heads = test_model. build_and_run(
        perlen, nstp, steady, recharge_by_period,
        hk=10.0, sy=0.15, strt=initial_strt,
        pumping_rate=-100.0,  # Test with some pumping
        trend_adjustment=trend_info['slope'],
        silent=False
    )
    
    if test_heads is None:
        print("  ✗ Test model failed!")
        return None, None, None, None
    
    print(f"  ✓ Test model successful!")
    print(f"    Simulated heads: {test_heads. min():.2f} - {test_heads.max():.2f} m")
    
    shutil.rmtree(test_ws, ignore_errors=True)
    
    # Calibration (train-period objective only -- see EnhancedCalibrator)
    calibrator = EnhancedCalibrator(
        config=config,
        perlen=perlen,
        nstp=nstp,
        steady=steady,
        recharge_by_period=recharge_by_period,
        observed_heads=observed_head_monthly,
        trend_info=trend_info,
        train_cutoff_idx=train_cutoff_idx,
    )
    
    best_params = calibrator.calibrate(maxiter=maxiter, popsize=popsize)
    
    if best_params is None:
        print("\n  ✗ Calibration failed")
        return None, None, None, None
    
    # Final model
    print("\n" + "="*70)
    print("     RUNNING FINAL CALIBRATED MODEL")
    print("="*70)
    
    final_ws = os.path. join(model_dir, 'final')
    if os. path.exists(final_ws):
        shutil.rmtree(final_ws, ignore_errors=True)
        time.sleep(0.1)
    os.makedirs(final_ws, exist_ok=True)
    
    final_config = EnhancedMODFLOWConfig(
        model_name='gw_final',
        model_ws=final_ws,
        exe_name=exe_name,
        nlay=2, nrow=15, ncol=15,
        delr=500.0, delc=500.0,
        top=reference_elevation, botm=[50.0, 0.0],
        well_row=7, well_col=7,
    )
    
    final_recharge = [r * best_params['rch_mult'] for r in recharge_by_period]
    
    final_model = EnhancedModflowModel(final_config)
    simulated_head_m = final_model. build_and_run(
        perlen, nstp, steady, final_recharge,
        hk=best_params['hk'], sy=best_params['sy'], strt=best_params['strt'],
        pumping_rate=best_params['pumping'],
        trend_adjustment=trend_info['slope'],
        silent=False
    )
    
    if simulated_head_m is None: 
        print("  ✗ Final model failed!")
        return None, None, None, None
    
    simulated_depth_ft = data_prep.convert_head_to_depth(simulated_head_m, reference_elevation)
    
    # Evaluate
    print("\n" + "="*70)
    print("     FINAL EVALUATION")
    print("="*70)
    
    n = min(len(simulated_depth_ft), len(observed_depth_monthly))
    simulated_depth_ft = simulated_depth_ft[:n]
    observed_depth_monthly = observed_depth_monthly[:n]
    period_dates = period_dates[:n]
    train_cutoff_idx = min(train_cutoff_idx, n)
    val_cutoff_idx = min(val_cutoff_idx, n)
    
    metrics = calculate_metrics(observed_depth_monthly, simulated_depth_ft)
    
    print("\n  Final Metrics (feet, ALL periods combined):")
    print("  " + "-"*40)
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    
    # LEAKAGE-AWARE EVALUATION: report train (fit) vs. val/test (genuinely
    # held-out, never seen by the calibration objective) metrics separately.
    # This is the split that should be quoted in the paper/rebuttal --
    # the combined-period number above mixes fit and generalization and
    # will look better than the model's true out-of-sample skill.
    split_metrics = {}
    split_slices = {
        'train': slice(0, train_cutoff_idx),
        'val': slice(train_cutoff_idx, val_cutoff_idx),
        'test': slice(val_cutoff_idx, n),
    }
    print("\n  Metrics by split (feet) -- val/test are genuinely held-out:")
    print("  " + "-"*40)
    for split_name, sl in split_slices.items():
        obs_split = observed_depth_monthly[sl]
        sim_split = simulated_depth_ft[sl]
        if len(obs_split) == 0:
            continue
        split_metrics[split_name] = calculate_metrics(obs_split, sim_split)
        print(f"\n    {split_name.upper()} ({len(obs_split)} periods):")
        for k, v in split_metrics[split_name].items():
            print(f"      {k}: {v:.4f}")
    
    # Plot and save
    plot_enhanced_results(
        period_dates, observed_depth_monthly, simulated_depth_ft,
        metrics, save_path=os.path.join(model_dir, 'enhanced_calibration_results.png')
    )
    
    results_df = pd.DataFrame({
        'date': period_dates,
        'observed_depth_ft': observed_depth_monthly,
        'simulated_depth_ft': simulated_depth_ft,
        'residual_ft': simulated_depth_ft - observed_depth_monthly,
        'split': (
            ['train'] * (train_cutoff_idx - 0) +
            ['val'] * (val_cutoff_idx - train_cutoff_idx) +
            ['test'] * (n - val_cutoff_idx)
        ),
    })
    results_df.to_csv(os.path.join(model_dir, 'enhanced_predictions.csv'), index=False)
    
    joblib.dump({
        'metrics': metrics,
        'split_metrics': split_metrics,
        'train_cutoff_idx': train_cutoff_idx,
        'val_cutoff_idx': val_cutoff_idx,
        'best_params':  best_params,
        'trend_info': trend_info,
        'calibration_results': calibrator.results,
    }, os.path. join(model_dir, 'enhanced_results.joblib'))
    
    print("\n" + "█"*70)
    print("   CALIBRATION COMPLETE!")
    print("█"*70)
    
    return simulated_depth_ft, observed_depth_monthly, metrics, best_params

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__": 
    
    simulated, observed, metrics, params = run_enhanced_modflow_pipeline(
        data_path=r'data/engineered/engineered_dataset.csv',
        model_dir=r'models/modflow_enhanced',
        target_col='water_level',
        date_col='date',
        train_end='2021-12-31',
        val_end='2023-12-31',
        exe_name=r'MF2005.1_12/MF2005.1_12/bin/mf2005.exe',
        recharge_fraction=0.15,
        reference_elevation=100.0,
        maxiter=80,
        popsize=15,
    )
    
    if params: 
        print("\n" + "="*50)
        print("FINAL CALIBRATED PARAMETERS")
        print("="*50)
        for k, v in params.items():
            print(f"  {k}: {v:.4f}")