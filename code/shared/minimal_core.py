"""INPUT: settings.py, hardware Excel and frozen daily/revenue CSVs.
OUTPUT: in-memory calculations for Table_MC_Summary and windows only.
No time-series resampling, inventory bootstrap, economic-grid or extra exports.
"""
from __future__ import annotations
import hashlib
from .settings import DEFINITIONS
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import triang
HERE = Path(__file__).resolve().parent
REVENUE_COLUMNS = {'ETC': 'revenue_USD_excluding_uncle_rewards', 'ETHW': 'subsidy_USD', 'RVN': 'scheduled_subsidy_USD', 'ERG': 'subsidy_USD'}

def window(frame: pd.DataFrame, start: str, end: str, columns=None) -> pd.DataFrame:
    result = frame.loc[start:end] if columns is None else frame.loc[start:end, columns]
    if not result.index.equals(pd.date_range(start, end)):
        raise ValueError(f'Incomplete daily window: {start} to {end}')
    if result.isna().to_numpy().any() or not np.isfinite(result.to_numpy(float)).all():
        raise ValueError(f'Missing/nonfinite observations: {start} to {end}')
    return result

def budget(revenue: np.ndarray, tariff: np.ndarray) -> np.ndarray:
    if np.any(np.asarray(tariff) <= 0):
        raise ValueError('Tariff must be positive')
    return np.asarray(revenue) / (24000000.0 * np.asarray(tariff))

DEFAULT_SOURCE = HERE.parents[1]

class MinimalModel:

    def __init__(self, source: Path=DEFAULT_SOURCE):
        self.source = Path(source).resolve()
        definitions = DEFINITIONS
        self.config = definitions['CONFIG']
        self.reference = definitions['FROZEN_REFERENCE']
        self.input_hashes = definitions['INPUT_CHECKSUMS']
        for (rel, expected) in self.input_hashes.items():
            if hashlib.sha256((self.source / rel).read_bytes()).hexdigest() != expected:
                raise ValueError('Input checksum mismatch: ' + rel)
        self.u = self.config['uq']
        self.d = self.config['dates']
        self.date = pd.Timestamp(self.d['last_pow'])
        self.daily = pd.read_csv(self.source / 'data/extension/inputs/crypto_daily.csv', parse_dates=['date_utc']).set_index('date_utc').sort_index()
        for (label, frame) in [('daily', self.daily)]:
            if not frame.index.is_unique:
                raise ValueError('Duplicate date: ' + label)
        self.revenue = pd.DataFrame()
        for (coin, column) in REVENUE_COLUMNS.items():
            frame = pd.read_csv(self.source / f'data/extension/inputs/{coin.lower()}_revenue.csv', parse_dates=['date_utc']).set_index('date_utc').sort_index()
            if not frame.index.is_unique:
                raise ValueError('Duplicate revenue date: ' + coin)
            self.revenue[coin] = frame[column] if self.revenue.empty else frame[column].reindex(self.revenue.index)
        self.stable_revenue = window(self.revenue, self.d['stable_start'], self.d['stable_end'])
        self.total_revenue = float(self.stable_revenue.mean().sum())
        self.h0 = float(self.daily.loc[self.date, 'ETH_hashrate_TH_s'])
        self.pre_hash = window(self.daily[['ETC_hashrate_TH_s']], self.d['pre_start'], self.d['last_pow'])
        self.post_hash = window(self.daily[['ETC_hashrate_TH_s']], self.d['stable_start'], self.d['stable_end'])
        self.hm = float(self.post_hash.iloc[:, 0].mean() - self.pre_hash.iloc[:, 0].mean())
        self.theta_pre = float(self.daily.loc[self.date, 'ETH_min_efficiency_MH_J'])
        self.theta_post = float(np.mean(self.post_hash.iloc[:, 0] / (1000 * budget(self.stable_revenue.ETC, 0.1))))
        columns = {'Miner_name': 'device', 'Type': 'type', 'Date of release': 'release', 'Hashing power (Mh/s)': 'hash_MH_s', 'Power (W)': 'power_W', 'Memory size in Gigabyte (GB)': 'memory_GB'}
        parts = []
        for filename in ['EthereumHardware.xlsx', 'EthereumHardwareGPU.xlsx']:
            frame = pd.read_excel(self.source / 'data/original' / filename).rename(columns=columns)
            frame['device'] = frame.device.str.strip()
            frame['type'] = frame.type.str.strip().str.lower()
            frame['release'] = pd.to_datetime(frame.release)
            frame['source_file'] = filename
            frame['source_excel_row'] = np.arange(len(frame)) + 2
            frame['efficiency'] = frame.hash_MH_s / frame.power_W
            parts.append(frame)
        self.raw = pd.concat(parts, ignore_index=True)
        duplicate = self.raw.duplicated(['device', 'hash_MH_s', 'power_W', 'memory_GB'], keep='first')
        self.hw = self.raw.loc[~duplicate].copy().reset_index(drop=True)
        if not self.hw.type.isin(['asic', 'gpu']).all():
            raise ValueError('Unknown hardware type')
        numbers = self.hw[['hash_MH_s', 'power_W', 'efficiency']].to_numpy(float)
        if not np.isfinite(numbers).all() or (numbers <= 0).any():
            raise ValueError('Invalid hardware hashrate/power')
        self.e = self.hw.efficiency.to_numpy()
        self.h = self.hw.hash_MH_s.to_numpy()
        self.asic = self.hw.type.eq('asic').to_numpy()
        self.release = (self.hw.release <= self.date).to_numpy()
        self.memory = self.hw.memory_GB.to_numpy()

    def classify(self, c: np.ndarray, lam: np.ndarray, memory_rule: str | None=None, theta_post: float | None=None) -> tuple:
        (c, lam) = np.broadcast_arrays(np.atleast_1d(c), np.atleast_1d(lam))
        if (c <= 0).any() or (lam <= 0).any():
            raise ValueError('Tariff and overhead must be positive')
        rule = self.u['memory_rule'] if memory_rule is None else memory_rule
        if rule not in ['strict', 'allow_5GB']:
            raise ValueError('Unknown memory rule')
        mem = self.memory > self.u['memory_GB'] if rule == 'strict' else self.memory >= self.u['memory_GB']
        wall = self.e[None, :] / lam[:, None]
        eligible = self.release[None, :] & mem[None, :] & (wall >= self.theta_pre * c[:, None] / 0.1)
        post = self.theta_post if theta_post is None else theta_post
        mig = eligible & (wall >= post * c[:, None] / 0.1)
        return (mig, eligible & ~mig)

    def weights(self, mask: np.ndarray, method: str, q: np.ndarray=0.25) -> np.ndarray:
        mask = np.atleast_2d(mask)
        q = np.broadcast_to(np.asarray(q, dtype=float), (len(mask),))
        if ((q < 0) | (q > 1)).any():
            raise ValueError('Target ASIC share outside [0, 1]')
        if method == 'equal_hash':
            base = np.ones(len(self.e))
            weights = mask * base[None, :]
            den = weights.sum(1)
            return np.divide(weights, den[:, None], out=np.zeros_like(weights, dtype=float), where=den[:, None] > 0)
        if method != 'type_share':
            raise ValueError('Unknown weighting method')
        na = (mask & self.asic).sum(1)
        ng = (mask & ~self.asic).sum(1)
        aq = np.where(na == 0, 0, np.where(ng == 0, 1, q))
        wg = np.divide(1 - aq, ng, out=np.zeros(len(mask)), where=ng > 0)
        wa = np.divide(aq, na, out=np.zeros(len(mask)), where=na > 0)
        return mask * np.where(self.asic[None, :], wa[:, None], wg[:, None])

    def outcome(self, wm: np.ndarray, wr: np.ndarray, c: np.ndarray, lam: np.ndarray, hm: np.ndarray | None=None, revenue: np.ndarray | None=None) -> pd.DataFrame:
        n = len(wm)
        hm = np.broadcast_to(np.asarray(self.hm if hm is None else hm), (n,))
        c = np.broadcast_to(c, (n,))
        lam = np.broadcast_to(lam, (n,))
        valid = (wm.sum(1) > 0) & (wr.sum(1) > 0) & (hm >= 0) & (hm <= self.h0)
        power = 0.001 * lam * (hm * (wm / self.e).sum(1) + (self.h0 - hm) * (wr / self.e).sum(1))
        power[~valid] = np.nan
        post = np.broadcast_to(budget(self.total_revenue if revenue is None else revenue, c), (n,)).copy()
        share = (hm * (wm * self.asic).sum(1) + (self.h0 - hm) * (wr * self.asic).sum(1)) / self.h0
        share[~valid] = np.nan
        if not np.isfinite(power[valid]).all() or (power[valid] <= 0).any():
            raise ValueError('Invalid power for a valid draw')
        for w in [wm, wr]:
            if not np.allclose(w[valid].sum(1), 1):
                raise ValueError('Cohort weights do not sum to one')
        return pd.DataFrame({'price_USD_kWh': c, 'overhead': lam, 'delta_h_TH_s': hm, 'pre_GW': power, 'post_budget_GW': post, 'saving_at_budget_pct': 100 * (1 - post / power), 'realized_ASIC_hash_share': share, 'valid': valid})

    def deterministic(self, method: str='equal_hash', q: float=0.25, c: float | None=None, lam: float | None=None, hm: float | None=None, revenue: float | None=None, theta_post: float | None=None) -> dict:
        c = self.u['price'] if c is None else c
        lam = self.u['overhead'] if lam is None else lam
        (mig, res) = self.classify(c, lam, theta_post=theta_post)
        out = self.outcome(self.weights(mig, method, q), self.weights(res, method, q), c, lam, hm, revenue).iloc[0].to_dict()
        return {**out, 'n_migrating': int(mig.sum()), 'n_residual': int(res.sum())}

    def simulate(self, spec: dict, index: int, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed + index)
        z = rng.standard_normal((n, 2))
        z[:, 1] = spec['rho'] * z[:, 0] + np.sqrt(1 - spec['rho'] ** 2) * z[:, 1]

        def triangular(u: np.ndarray, values: list) -> np.ndarray:
            (lo, mode, hi) = values
            return triang.ppf(u, (mode - lo) / (hi - lo), loc=lo, scale=hi - lo)
        c = triangular(ndtr(z[:, 0]), self.u['price_triangular'])
        lam = triangular(ndtr(z[:, 1]), self.u['overhead_triangular'])
        if spec.get('fixed_econ'):
            c[:] = self.u['price']
            lam[:] = self.u['overhead']
        q = np.zeros(n) if spec['beta'] is None else rng.beta(*spec['beta'], size=n)
        (mig, res) = self.classify(c, lam, spec.get('memory_rule'))
        method = 'equal_hash' if spec['beta'] is None else 'type_share'
        wm = self.weights(mig, method, q)
        wr = self.weights(res, method, q)
        if spec['kappa'] is not None:
            for w in [wm, wr]:
                gamma = rng.gamma(spec['kappa'] * w, 1.0)
                den = gamma.sum(1)
                w[:] = np.divide(gamma, den[:, None], out=np.zeros_like(gamma), where=den[:, None] > 0)
        if np.any(wm[~mig]) or np.any(wr[~res]):
            raise ValueError('Weight assigned to ineligible model')
        (hm, rev) = (None, None)
        out = self.outcome(wm, wr, c, lam, hm, rev)
        out['target_ASIC_share_mixed'] = q
        out['n_migrating'] = mig.sum(1)
        out['n_residual'] = res.sum(1)
        return out

def summary(draws: pd.DataFrame, key: str, label: str) -> dict:
    valid = draws.loc[draws.valid]
    row = {'scenario': label, 'key': key, 'draws': len(draws), 'valid_draws': len(valid), 'invalid_pct': 100 * (1 - len(valid) / len(draws))}
    for col in ['pre_GW', 'post_budget_GW', 'saving_at_budget_pct', 'realized_ASIC_hash_share']:
        vals = valid[col].to_numpy()
        qq = np.quantile(vals, [0.025, 0.5, 0.975]) if len(vals) else [np.nan] * 3
        for (suffix, value) in zip(['p025', 'median', 'p975'], qq):
            row[col + '_' + suffix] = value
    return row

def reference_check(model: MinimalModel, frame: pd.DataFrame, target: str, keys: list, columns: list) -> dict:
    expected = pd.read_csv(StringIO(model.reference[target])).set_index(keys).sort_index()
    current = frame.set_index(keys).sort_index()
    expected = expected.loc[current.index]
    for col in columns:
        np.testing.assert_allclose(current[col].astype(float), expected[col].astype(float), rtol=2e-08, atol=2e-08, equal_nan=True)
    return {'target': target, 'rows_checked': len(current), 'passed': True, 'columns': columns}
