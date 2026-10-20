"""
Comprehensive evaluation of 5 clustering algorithms in place of K-Means:
1. Gaussian Mixture (GMM)
2. Mean Shift
3. DBSCAN
4. Hierarchical (Agglomerative - Ward)
5. Spectral Clustering

Evaluated across two primary paradigms:
Mode A: Used to initialize the Hidden Markov Model (replacing kmeans_em_init).
Mode B: Used directly as the Regime Classifier (bypassing the HMM temporal transitions).

Computes full In-Sample and Out-of-Sample (Walk-Forward across 27 folds) metrics.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
from datetime import datetime

_SRC = os.path.abspath('src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import (
    KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering, estimate_bandwidth
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from hmmlearn.hmm import GaussianHMM

from regime_detector_v2 import (
    fetch_live_market_data, fetch_live_sector_data, engineer_features,
    select_features_for_hmm, optimize_sector_weights, smooth_regimes_causal,
    online_forward_decode, new_stoploss_state, stoploss_overlay_step,
    STOP_LOSS_DD, STOP_REENTRY_MA, STOP_REENTRY_TYPE, STOP_COOLDOWN_DAYS,
    TC_TOTAL, WF_TRAIN_YEARS, WF_TEST_MONTHS
)

# ══════════════════════════════════════════════════════════════════════
# 1. CLUSTER EXTRACTION WRAPPERS (4 STATES GUARANTEE)
# ══════════════════════════════════════════════════════════════════════

def extract_clusters_kmeans(X, n_states=4, seed=42):
    km = KMeans(n_clusters=n_states, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_
    return labels, centers, km

def extract_clusters_gmm(X, n_states=4, seed=42):
    gmm = GaussianMixture(n_components=n_states, covariance_type='diag', random_state=seed, n_init=5).fit(X)
    labels = gmm.predict(X)
    centers = gmm.means_
    return labels, centers, gmm

def extract_clusters_hierarchical(X, n_states=4, seed=42):
    agg = AgglomerativeClustering(n_clusters=n_states, linkage='ward').fit(X)
    labels = agg.labels_
    centers = np.array([X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else np.zeros(X.shape[1]) for k in range(n_states)])
    return labels, centers, agg

def extract_clusters_spectral(X, n_states=4, seed=42):
    spec = SpectralClustering(
        n_clusters=n_states, affinity='nearest_neighbors', n_neighbors=15,
        n_init=3, random_state=seed, assign_labels='kmeans'
    ).fit(X)
    labels = spec.labels_
    centers = np.array([X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else np.zeros(X.shape[1]) for k in range(n_states)])
    return labels, centers, spec

def extract_clusters_meanshift(X, n_states=4, seed=42):
    # Search bandwidth for ~4 clusters, else take top 4 modes
    bw = estimate_bandwidth(X, quantile=0.08, n_samples=min(500, len(X)))
    if bw is None or bw <= 0:
        bw = 3.2
    ms = MeanShift(bandwidth=bw, bin_seeding=True, cluster_all=True).fit(X)
    raw_labels = ms.labels_
    unique, counts = np.unique(raw_labels, return_counts=True)
    
    # Sort clusters by popularity
    sorted_clusters = [u for _, u in sorted(zip(counts, unique), reverse=True)]
    top_4 = sorted_clusters[:n_states]
    
    # If fewer than 4 clusters found, split largest with KMeans fallback
    if len(top_4) < n_states:
        return extract_clusters_kmeans(X, n_states=n_states, seed=seed)
    
    # Compute centers for top 4
    top_centers = np.array([X[raw_labels == c].mean(axis=0) for c in top_4])
    
    # Reassign all points to nearest top 4 center
    dists = np.linalg.norm(X[:, None, :] - top_centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    centers = np.array([X[labels == k].mean(axis=0) for k in range(n_states)])
    return labels, centers, ms

def extract_clusters_dbscan(X, n_states=4, seed=42):
    # Calibrate eps for ~4 clusters on 11-d quantile data
    db = DBSCAN(eps=2.5, min_samples=20).fit(X)
    raw_labels = db.labels_
    
    # Exclude noise (-1) to find genuine core clusters
    core_labels = raw_labels[raw_labels >= 0]
    if len(core_labels) == 0:
        return extract_clusters_kmeans(X, n_states=n_states, seed=seed)
        
    unique, counts = np.unique(core_labels, return_counts=True)
    sorted_clusters = [u for _, u in sorted(zip(counts, unique), reverse=True)]
    top_4 = sorted_clusters[:n_states]
    
    if len(top_4) < n_states:
        return extract_clusters_kmeans(X, n_states=n_states, seed=seed)
        
    top_centers = np.array([X[raw_labels == c].mean(axis=0) for c in top_4])
    # Reassign noise and non-top points to nearest top 4 center
    dists = np.linalg.norm(X[:, None, :] - top_centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    centers = np.array([X[labels == k].mean(axis=0) for k in range(n_states)])
    return labels, centers, db

CLUSTER_METHODS = {
    'K-Means': extract_clusters_kmeans,
    'Gaussian Mixture': extract_clusters_gmm,
    'Mean Shift': extract_clusters_meanshift,
    'DBSCAN': extract_clusters_dbscan,
    'Hierarchical': extract_clusters_hierarchical,
    'Spectral': extract_clusters_spectral,
}

# ══════════════════════════════════════════════════════════════════════
# 2. HMM EM INITIALIZATION FROM CLUSTER RESULTS
# ══════════════════════════════════════════════════════════════════════

def build_hmm_em_init(X_scaled, labels, centers, n_states=4):
    n_samples, n_features = X_scaled.shape
    init_means = centers
    init_covars = np.zeros((n_states, n_features))
    for k in range(n_states):
        pts = X_scaled[labels == k]
        if len(pts) > 1:
            cov = np.var(pts, axis=0) + 1e-3
        else:
            cov = np.ones(n_features)
        init_covars[k] = np.maximum(cov, 1e-3)
        
    init_transmat = np.full((n_states, n_states), 0.02 / (n_states - 1))
    np.fill_diagonal(init_transmat, 0.98)
    init_startprob = np.ones(n_states) / n_states
    return init_startprob, init_transmat, init_means, init_covars

def train_custom_hmm(X_scaled, cluster_func, n_states=4, n_iter=150, seed=42):
    labels, centers, _ = cluster_func(X_scaled, n_states=n_states, seed=seed)
    sp, tm, mu, cv = build_hmm_em_init(X_scaled, labels, centers, n_states=n_states)
    
    m = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        tol=1e-4,
        min_covar=1e-3,
        random_state=seed,
        init_params='',
        verbose=False
    )
    m.startprob_ = sp
    m.transmat_  = tm
    m.means_     = mu
    m.covars_    = cv
    m.fit(X_scaled)
    return m

# ══════════════════════════════════════════════════════════════════════
# 3. REGIME LABELLING LOGIC (Statistical Profile)
# ══════════════════════════════════════════════════════════════════════

def label_clusters_economically(labels, returns, drawdown, n_states=4):
    state_stats = {}
    for s in range(n_states):
        m = (labels == s)
        state_stats[s] = {
            'ann_ret': float(returns[m].mean() * 252 * 100) if m.sum() > 0 else 0.0,
            'ann_vol': float(returns[m].std() * np.sqrt(252) * 100) if m.sum() > 0 else 0.0,
            'drawdown': float(drawdown[m].mean()) if m.sum() > 0 else 0.0,
            'count': int(m.sum())
        }
    # Deepest DD -> Bear
    s_bear = min(range(n_states), key=lambda s: state_stats[s]['drawdown'])
    rem = [s for s in range(n_states) if s != s_bear]
    # Highest return -> Bull
    s_bull = max(rem, key=lambda s: state_stats[s]['ann_ret'])
    rem2 = [s for s in rem if s != s_bull]
    # Lower vol -> Sideways, Higher vol -> HighVol
    rem2_sorted = sorted(rem2, key=lambda s: state_stats[s]['ann_vol'])
    s_sideways = rem2_sorted[0]
    s_highvol = rem2_sorted[1]
    
    lmap = {
        s_bull: 'Bull',
        s_bear: 'Bear',
        s_highvol: 'HighVol',
        s_sideways: 'Sideways'
    }
    return lmap, state_stats

# ══════════════════════════════════════════════════════════════════════
# 4. BACKTEST SIMULATION (IN-SAMPLE)
# ══════════════════════════════════════════════════════════════════════

def run_is_backtest(regimes, df, df_sec, sector_names, stop_dd=STOP_LOSS_DD):
    regime_dates = pd.Index(list(regimes.keys()))
    common_idx = regime_dates.intersection(df_sec.index).intersection(df.index)
    rf_daily = 0.055 / 252.0
    
    # 1. Compute optimal mix per regime on full period
    sector_mix = {}
    for reg in ['Bull', 'Bear', 'HighVol', 'Sideways']:
        reg_dates = [idx for idx in common_idx if regimes.get(idx) == reg]
        if len(reg_dates) < 30:
            sector_mix[reg] = {s: round(1.0/len(sector_names), 3) for s in sector_names}
        else:
            sec_rets = df_sec.loc[reg_dates, sector_names].values
            mean_ret = sec_rets.mean(axis=0)
            cov_mat  = np.cov(sec_rets, rowvar=False) + np.eye(len(sector_names)) * 1e-6
            sector_mix[reg] = optimize_sector_weights(mean_ret, cov_mat, reg, sector_names, rf_daily)

    # 2. Run simulation with T+1 execution and stoploss
    reg_series = pd.Series([regimes[d] for d in common_idx], index=common_idx)
    reg_signals = reg_series.shift(1).fillna(reg_series.iloc[0])
    
    nifty_prices = df.loc[common_idx, 'NIFTY']
    nifty_ma = nifty_prices.rolling(STOP_REENTRY_MA, min_periods=1).mean()
    nifty_above_ma = nifty_prices > nifty_ma
    daily_repo = pd.Series(rf_daily, index=common_idx)
    
    stop_state = new_stoploss_state()
    prev_reg = reg_signals.iloc[0]
    returns = []
    switches = 0
    
    for i in range(len(common_idx)):
        dt = common_idx[i]
        reg = reg_signals.iloc[i]
        if reg != prev_reg:
            switches += 1
            tc = TC_TOTAL
        else:
            tc = 0.0
            
        w = sector_mix.get(reg, {})
        port_r = sum(w.get(s, 0.0) * float(df_sec.loc[dt, s]) for s in sector_names if s in df_sec.columns)
        base_r = port_r - tc
        
        daily_r = stoploss_overlay_step(
            stop_state, base_r, float(daily_repo.iloc[i]), bool(nifty_above_ma.iloc[i]),
            stop_dd=stop_dd, cooldown_days=STOP_COOLDOWN_DAYS
        )
        returns.append(daily_r)
        prev_reg = reg
        
    rets = np.array(returns)
    cum = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    
    ann_ret = np.mean(rets) * 252 * 100
    ann_vol = np.std(rets) * np.sqrt(252) * 100
    sharpe  = (ann_ret - 5.5) / ann_vol if ann_vol > 0 else 0.0
    downside_vol = np.std(rets[rets < 0]) * np.sqrt(252) * 100 if np.sum(rets < 0) > 0 else 1.0
    sortino = (ann_ret - 5.5) / downside_vol if downside_vol > 0 else 0.0
    max_dd  = np.min(dd) * 100
    calmar  = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    win_rate = (rets > 0).mean() * 100
    
    return {
        'ann_ret': round(ann_ret, 2),
        'ann_vol': round(ann_vol, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd, 2),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 2),
        'switches': switches,
        'stops': stop_state['stops'],
        'days_cash': stop_state['days_cash']
    }

# ══════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD OOS EVALUATION
# ══════════════════════════════════════════════════════════════════════

def run_wf_evaluation(mode, method_name, cluster_func, feat, df, df_sec, sector_names, n_states=4):
    all_dates = feat.index
    start_date = all_dates[0]
    train_td = pd.DateOffset(years=WF_TRAIN_YEARS)
    test_td  = pd.DateOffset(months=WF_TEST_MONTHS)
    
    fold_start = start_date + train_td
    stop_state = new_stoploss_state()
    all_oos_returns = []
    
    nifty_prices = df['NIFTY']
    nifty_ma = nifty_prices.rolling(STOP_REENTRY_MA, min_periods=1).mean()
    nifty_above_ma = nifty_prices > nifty_ma
    rf_daily = 0.055 / 252.0
    daily_repo = pd.Series(rf_daily, index=df.index)
    
    prev_regime = None
    folds_count = 0
    
    while fold_start < all_dates[-1]:
        fold_end = min(fold_start + test_td, all_dates[-1])
        train_mask = (all_dates >= start_date) & (all_dates < fold_start)
        test_mask  = (all_dates >= fold_start) & (all_dates < fold_end)
        
        if train_mask.sum() < 200 or test_mask.sum() < 25:
            fold_start = fold_end
            continue
            
        folds_count += 1
        X_train = select_features_for_hmm(feat.loc[train_mask]).values
        X_test  = select_features_for_hmm(feat.loc[test_mask]).values
        
        scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X_train)), random_state=42)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        
        train_dates = all_dates[train_mask]
        train_ret   = df.loc[train_dates, 'Returns'].values
        train_dd    = feat.loc[train_dates, 'drawdown'].values
        
        if mode == 'HMM_INIT':
            # Mode A: Cluster algorithm initializes HMM
            model = train_custom_hmm(X_train_s, cluster_func, n_states=n_states, n_iter=100)
            _, states_train = model.decode(X_train_s, algorithm='viterbi')
            lmap, _ = label_clusters_economically(states_train, train_ret, train_dd, n_states=n_states)
            states_test = online_forward_decode(model, X_test_s)
            oos_labels = [lmap.get(s, 'Sideways') for s in states_test]
            train_labels = [lmap.get(s, 'Sideways') for s in states_train]
        else:
            # Mode B: Cluster algorithm directly classifies regimes
            labels_train, centers_train, fitted_model = cluster_func(X_train_s, n_states=n_states)
            lmap, _ = label_clusters_economically(labels_train, train_ret, train_dd, n_states=n_states)
            
            # Predict OOS
            if hasattr(fitted_model, 'predict'):
                try:
                    labels_test = fitted_model.predict(X_test_s)
                except Exception:
                    dists = np.linalg.norm(X_test_s[:, None, :] - centers_train[None, :, :], axis=2)
                    labels_test = np.argmin(dists, axis=1)
            else:
                dists = np.linalg.norm(X_test_s[:, None, :] - centers_train[None, :, :], axis=2)
                labels_test = np.argmin(dists, axis=1)
                
            oos_labels = [lmap.get(s, 'Sideways') for s in labels_test]
            train_labels = [lmap.get(s, 'Sideways') for s in labels_train]
            
        oos_labels = smooth_regimes_causal(oos_labels)
        
        # Optimize weights on training data
        fold_sector_mix = {}
        for reg in ['Bull', 'Bear', 'HighVol', 'Sideways']:
            reg_mask = [i for i, l in enumerate(train_labels) if l == reg]
            if len(reg_mask) < 30:
                fold_sector_mix[reg] = {s: round(1.0/len(sector_names), 3) for s in sector_names}
            else:
                t_dates = train_dates[reg_mask]
                common_dates = t_dates.intersection(df_sec.index)
                if len(common_dates) < 30:
                    fold_sector_mix[reg] = {s: round(1.0/len(sector_names), 3) for s in sector_names}
                else:
                    sec_rets = df_sec.loc[common_dates, sector_names].values
                    mean_ret = sec_rets.mean(axis=0)
                    cov_mat  = np.cov(sec_rets, rowvar=False) + np.eye(len(sector_names)) * 1e-6
                    fold_sector_mix[reg] = optimize_sector_weights(mean_ret, cov_mat, reg, sector_names, rf_daily)

        # Simulate OOS test fold (T+1 execution)
        oos_idx = all_dates[test_mask]
        if prev_regime is None:
            prev_regime = oos_labels[0]
            
        for i in range(len(oos_idx) - 1):
            reg_signal = oos_labels[i]
            dt_trade   = oos_idx[i + 1]
            w = fold_sector_mix.get(reg_signal, {})
            if dt_trade in df_sec.index:
                port_daily = sum(w.get(s, 0.0) * float(df_sec.loc[dt_trade, s]) for s in sector_names if s in df_sec.columns)
            else:
                port_daily = float(df.loc[dt_trade, 'Returns'])
                
            tc = TC_TOTAL if (reg_signal != prev_regime) else 0.0
            base_daily = port_daily - tc
            
            daily = stoploss_overlay_step(
                stop_state, base_daily, float(daily_repo.loc[dt_trade]),
                bool(nifty_above_ma.loc[dt_trade]), stop_dd=STOP_LOSS_DD,
                cooldown_days=STOP_COOLDOWN_DAYS
            )
            all_oos_returns.append(daily)
            prev_regime = reg_signal
            
        fold_start = fold_end
        
    rets = np.array(all_oos_returns)
    cum = np.exp(np.cumsum(rets))
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    
    ann_ret = np.mean(rets) * 252 * 100
    ann_vol = np.std(rets) * np.sqrt(252) * 100
    sharpe  = (ann_ret - 5.5) / ann_vol if ann_vol > 0 else 0.0
    downside_vol = np.std(rets[rets < 0]) * np.sqrt(252) * 100 if np.sum(rets < 0) > 0 else 1.0
    sortino = (ann_ret - 5.5) / downside_vol if downside_vol > 0 else 0.0
    max_dd  = np.min(dd) * 100
    calmar  = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    win_rate = (rets > 0).mean() * 100
    
    return {
        'folds': folds_count,
        'ann_ret': round(ann_ret, 2),
        'ann_vol': round(ann_vol, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd, 2),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 2),
        'stops': stop_state['stops'],
        'days_cash': stop_state['days_cash']
    }

# ══════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION SCRIPT
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CLUSTERING ALGORITHMS COMPARISON BENCHMARK (INDIA MARKET REGIME)")
    print("  Algorithms Tested: K-Means, Gaussian Mixture, Mean Shift, DBSCAN, Hierarchical, Spectral")
    print("=" * 70)
    
    df = fetch_live_market_data(start_date="2015-01-01")
    df_sec = fetch_live_sector_data(start_date="2015-01-01")
    feat = engineer_features(df)
    X = select_features_for_hmm(feat).values
    
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    X_s = scaler.fit_transform(X)
    
    common_idx = feat.index
    sector_names = ['BANKBEES', 'ITBEES', 'PHARMABEES', 'AUTOBEES', 'METALIETF', 'MOREALTY', 'CPSEETF', 'INFRABEES']
    
    results_summary = []
    
    # ── PART 1: CLUSTER QUALITY & DIRECT CLUSTERING REGIME ENGINE ──
    print("\n" + "="*50)
    print("PART 1: Direct Clustering as Regime Classifier")
    print("="*50)
    
    for name, func in CLUSTER_METHODS.items():
        t0 = time.time()
        labels, centers, _ = func(X_s, n_states=4, seed=42)
        fit_time = time.time() - t0
        
        # Cluster geometry metrics
        sil = silhouette_score(X_s, labels)
        db = davies_bouldin_score(X_s, labels)
        ch = calinski_harabasz_score(X_s, labels)
        
        # Label economically
        lmap, stats = label_clusters_economically(labels, df.loc[common_idx, 'Returns'].values, feat['drawdown'].values, n_states=4)
        regimes_dict = {d: lmap.get(labels[i], 'Sideways') for i, d in enumerate(common_idx)}
        
        # In-Sample Backtest
        is_metrics = run_is_backtest(regimes_dict, df, df_sec, sector_names)
        
        # Out-of-Sample Walk-Forward Backtest
        oos_metrics = run_wf_evaluation('DIRECT_CLUSTER', name, func, feat, df, df_sec, sector_names, n_states=4)
        
        entry = {
            'Paradigm': 'Direct Clustering (No HMM)',
            'Algorithm': name,
            'Fit_Time_s': round(fit_time, 3),
            'Silhouette': round(sil, 4),
            'Davies_Bouldin': round(db, 4),
            'Calinski_Harabasz': round(ch, 1),
            'IS_AnnRet_pct': is_metrics['ann_ret'],
            'IS_Sharpe': is_metrics['sharpe'],
            'IS_Sortino': is_metrics['sortino'],
            'IS_MaxDD_pct': is_metrics['max_dd'],
            'IS_Calmar': is_metrics['calmar'],
            'IS_Switches': is_metrics['switches'],
            'OOS_AnnRet_pct': oos_metrics['ann_ret'],
            'OOS_Sharpe': oos_metrics['sharpe'],
            'OOS_Sortino': oos_metrics['sortino'],
            'OOS_MaxDD_pct': oos_metrics['max_dd'],
            'OOS_Calmar': oos_metrics['calmar'],
            'OOS_Stops': oos_metrics['stops'],
            'OOS_Days_Cash': oos_metrics['days_cash'],
        }
        results_summary.append(entry)
        print(f"✓ Completed {name:<18} | IS Sharpe: {is_metrics['sharpe']:>6.3f} | OOS Sharpe: {oos_metrics['sharpe']:>6.3f} | OOS Ret: {oos_metrics['ann_ret']:>6.2f}% | OOS MaxDD: {oos_metrics['max_dd']:>6.2f}%")

    # ── PART 2: CLUSTER ALGORITHMS FOR HMM EM INITIALIZATION ──
    print("\n" + "="*50)
    print("PART 2: HMM Initialized with Each Clustering Algorithm")
    print("="*50)
    
    for name, func in CLUSTER_METHODS.items():
        t0 = time.time()
        # Train HMM initialized with this cluster algorithm
        model = train_custom_hmm(X_s, func, n_states=4, n_iter=150, seed=42)
        fit_time = time.time() - t0
        
        ll = model.score(X_s)
        _, states_is = model.decode(X_s, algorithm='viterbi')
        lmap, _ = label_clusters_economically(states_is, df.loc[common_idx, 'Returns'].values, feat['drawdown'].values, n_states=4)
        regimes_dict = {d: lmap.get(states_is[i], 'Sideways') for i, d in enumerate(common_idx)}
        
        # In-Sample Backtest
        is_metrics = run_is_backtest(regimes_dict, df, df_sec, sector_names)
        
        # Out-of-Sample Walk-Forward Backtest
        oos_metrics = run_wf_evaluation('HMM_INIT', name, func, feat, df, df_sec, sector_names, n_states=4)
        
        entry = {
            'Paradigm': 'HMM EM Initializer',
            'Algorithm': f"HMM + {name}",
            'Fit_Time_s': round(fit_time, 3),
            'Silhouette': round(silhouette_score(X_s, states_is), 4),
            'Davies_Bouldin': round(davies_bouldin_score(X_s, states_is), 4),
            'Calinski_Harabasz': round(calinski_harabasz_score(X_s, states_is), 1),
            'IS_AnnRet_pct': is_metrics['ann_ret'],
            'IS_Sharpe': is_metrics['sharpe'],
            'IS_Sortino': is_metrics['sortino'],
            'IS_MaxDD_pct': is_metrics['max_dd'],
            'IS_Calmar': is_metrics['calmar'],
            'IS_Switches': is_metrics['switches'],
            'OOS_AnnRet_pct': oos_metrics['ann_ret'],
            'OOS_Sharpe': oos_metrics['sharpe'],
            'OOS_Sortino': oos_metrics['sortino'],
            'OOS_MaxDD_pct': oos_metrics['max_dd'],
            'OOS_Calmar': oos_metrics['calmar'],
            'OOS_Stops': oos_metrics['stops'],
            'OOS_Days_Cash': oos_metrics['days_cash'],
        }
        results_summary.append(entry)
        print(f"✓ Completed HMM + {name:<12} | IS Sharpe: {is_metrics['sharpe']:>6.3f} | OOS Sharpe: {oos_metrics['sharpe']:>6.3f} | OOS Ret: {oos_metrics['ann_ret']:>6.2f}% | OOS MaxDD: {oos_metrics['max_dd']:>6.2f}%")

    # Save to CSV and JSON
    res_df = pd.DataFrame(results_summary)
    res_df.to_csv('data/clustering_comparison_scorecard.csv', index=False)
    with open('data/clustering_comparison_scorecard.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    print("\n✓ Full Scorecard successfully saved to:")
    print("  - data/clustering_comparison_scorecard.csv")
    print("  - data/clustering_comparison_scorecard.json")

if __name__ == '__main__':
    main()
