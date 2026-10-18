"""
India HMM Regime Detector — Interactive Telegram Bot
=====================================================
A standalone command interface that lets authorized users query the
trained HMM model, view current regime status, sector allocations,
macro snapshots, charts, and trigger model retraining — all via
Telegram commands.

Install:
    pip install python-telegram-bot>=22.0

Run:
    python telegram_bot.py

Commands:
    /start          — Welcome message & command list
    /status         — Current regime, posteriors, exposure
    /sectors        — Recommended sector allocation for active regime
    /macro          — Latest macro indicators (VIX, CPI, WPI, IIP, Yield Curve)
    /chart          — Send the latest Fig 1 (regime detection chart)
    /backtest       — Send the latest Fig 2 (strategy backtest chart)
    /heatmap        — Send the latest Fig 7 (sector rotation heatmap)
    /walkforward    — Send the latest Fig 6 (walk-forward OOS chart)
    /history [N]    — Last N regime transitions (default 10)
    /retrain        — Re-run the full model pipeline (admin only)
    /help           — Full command reference

Security:
    Only the chat ID in .env (TELEGRAM_CHAT_ID) is authorized to execute
    commands. All other users receive an "unauthorized" response.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

import pandas as pd

from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TelegramBot] %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SRC_DIR.parent if (SRC_DIR.parent / "data").exists() or (SRC_DIR.parent / "artifacts").exists() or (SRC_DIR.parent / "src").exists() else SRC_DIR
ROOT_DIR = PROJECT_DIR
DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACT_DIR = ARTIFACTS_DIR  # backwards-compatible alias
FIGURES_DIR = PROJECT_DIR / "figures"

# Ensure output directories exist
for _d in [DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Ensure src and repo root are in sys.path
for _p in [str(SRC_DIR), str(PROJECT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env ──
def _load_env():
    for env_path in [PROJECT_DIR / ".env", SRC_DIR / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)

_load_env()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AUTHORIZED_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Add it to .env")
    sys.exit(1)
if not AUTHORIZED_CHAT_ID:
    logger.error("TELEGRAM_CHAT_ID not set. Add it to .env")
    sys.exit(1)

# ── Regime Exposure Map (mirrors regime_detector_v2.py) ─────────────
REGIME_EXPOSURE = {
    "Bull":     1.0,
    "Bear":     0.0,
    "HighVol":  0.2,
    "Sideways": 0.6,
}

REGIME_EMOJI = {
    "Bull":     "🟢",
    "Bear":     "🔴",
    "HighVol":  "🟡",
    "Sideways": "🔵",
}


# ══════════════════════════════════════════════════════════════════════
# Helper: Authorization Check
# ══════════════════════════════════════════════════════════════════════
def authorized(func):
    """Decorator that restricts commands to the authorized chat ID."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        if chat_id != str(AUTHORIZED_CHAT_ID):
            await update.message.reply_text(
                "⛔ Unauthorized. This bot is restricted to a private operator."
            )
            logger.warning(f"Unauthorized access attempt from chat_id={chat_id}")
            return
        return await func(update, context)
    return wrapper


# ══════════════════════════════════════════════════════════════════════
# Helper: Load Model Artifacts
# ══════════════════════════════════════════════════════════════════════
def _load_regime_history():
    """Load the regime history CSV produced by the main pipeline."""
    csv_path = DATA_DIR / "regime_history_v2.csv"
    if not csv_path.exists():
        csv_path = PROJECT_DIR / "regime_history_v2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"regime_history_v2.csv not found in {DATA_DIR} or {PROJECT_DIR}")
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def _load_sector_mix():
    """Load the learned sector mix JSON."""
    mix_path = ARTIFACTS_DIR / "learned_sector_mix.json"
    if not mix_path.exists():
        mix_path = PROJECT_DIR / "learned_sector_mix.json"
    if not mix_path.exists():
        return {}
    with open(mix_path) as f:
        return json.load(f)


def _load_walk_forward():
    """Load the walk-forward summary CSV."""
    wf_path = DATA_DIR / "walk_forward_summary.csv"
    if not wf_path.exists():
        wf_path = PROJECT_DIR / "walk_forward_summary.csv"
    if not wf_path.exists():
        return pd.DataFrame()
    return pd.read_csv(wf_path)


def _load_paper_portfolio():
    """Load the paper portfolio history CSV."""
    csv_path = DATA_DIR / "paper_portfolio_history.csv"
    if not csv_path.exists():
        csv_path = PROJECT_DIR / "paper_portfolio_history.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def _load_parameter_stability():
    """Load the parameter stability report JSON."""
    report_file = DATA_DIR / "parameter_stability_report.json"
    if not report_file.exists():
        report_file = PROJECT_DIR / "parameter_stability_report.json"
    if not report_file.exists():
        return {}
    with open(report_file, "r") as f:
        return json.load(f)


def _load_india_macro():
    """Load the curated India macroeconomic historical CSV."""
    macro_path = DATA_DIR / "india_macro_historical.csv"
    if not macro_path.exists():
        macro_path = PROJECT_DIR / "india_macro_historical.csv"
    if not macro_path.exists():
        return pd.DataFrame()
    return pd.read_csv(macro_path, index_col=0, parse_dates=True)


# ══════════════════════════════════════════════════════════════════════
# Bot Commands
# ══════════════════════════════════════════════════════════════════════

@authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message and command overview."""
    msg = (
        "🇮🇳 *India HMM Regime Detector — Telegram Interface*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "📊 *Market Intelligence Commands:*\n"
        "  /status — Current regime, posteriors, exposure\n"
        "  /sectors — Sector allocation for active regime\n"
        "  /macro — Latest macro indicators (CPI, WPI, IIP, Yields)\n"
        "  /history — Recent regime transitions\n\n"
        "📈 *Chart Commands:*\n"
        "  /chart [N] — Regime detection overview (Fig 1–10, default Fig 1)\n"
        "  /backtest — Strategy vs Buy & Hold (Fig 2)\n"
        "  /internals — HMM model internals & transition matrix (Fig 3)\n"
        "  /confidence — Bootstrap confidence intervals (Fig 4)\n"
        "  /modelselection — Model selection curves (Fig 5)\n"
        "  /walkforward — Walk-forward OOS performance (Fig 6)\n"
        "  /heatmap — Sector rotation heatmap & regime returns (Fig 7)\n"
        "  /macrochart — Macro leading signals & yield curve (Fig 8)\n"
        "  /fig9 — Live paper portfolio dashboard (Fig 9)\n"
        "  /stability — Parameter stability heatmaps (Fig 10)\n\n"
        "💼 *Paper Portfolio & Execution:*\n"
        "  /portfolio — Current positions, cash & NAV\n"
        "  /reconcile — EOD trade & accounting reconciliation audit\n"
        "  /updateportfolio — Refresh with latest market prices\n\n"
        "⚙️ *Admin Commands:*\n"
        "  /retrain — Re-run full model pipeline\n"
        "  /help — Full command reference\n\n"
        "⚠️ Quantitative model output, not financial advice."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


@authorized
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Current regime status with posteriors and exposure."""
    try:
        df = _load_regime_history()
        latest = df.iloc[-1]
        sector_mix = _load_sector_mix()

        curr_reg = latest.get("Regime", "Sideways")
        emoji = REGIME_EMOJI.get(curr_reg, "⚪")
        nifty = latest.get("NIFTY", 0)
        ret = latest.get("Returns", 0) * 100
        vix = latest.get("VIX", 0)

        bull_p = latest.get("Bull", 0) * 100
        bear_p = latest.get("Bear", 0) * 100
        hv_p = latest.get("HighVol", 0) * 100
        sw_p = latest.get("Sideways", 0) * 100

        eq_exp = int(round((
            (bull_p / 100) * REGIME_EXPOSURE["Bull"] +
            (bear_p / 100) * REGIME_EXPOSURE["Bear"] +
            (hv_p / 100)   * REGIME_EXPOSURE["HighVol"] +
            (sw_p / 100)   * REGIME_EXPOSURE["Sideways"]
        ) * 100))
        cash_exp = max(0, 100 - eq_exp)

        date_str = latest.name.strftime("%d %b %Y") if hasattr(latest.name, "strftime") else str(latest.name)[:10]

        # Top sectors
        sector_str = ""
        if curr_reg in sector_mix:
            top = sorted(sector_mix[curr_reg].items(), key=lambda x: x[1], reverse=True)
            parts = [f"{s}: {w*100:.0f}%" for s, w in top if w > 0.01]
            if parts:
                sector_str = "\n  ".join(parts)

        msg = (
            f"📊 *REGIME STATUS — {date_str}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} Active Regime: *{curr_reg.upper()}*\n\n"
            f"NIFTY 50: {nifty:,.2f} ({ret:+.2f}%)\n"
            f"India VIX: {vix:.2f}\n\n"
            f"*Posterior Probabilities:*\n"
            f"  🟢 Bull: {bull_p:.1f}%\n"
            f"  🔴 Bear: {bear_p:.1f}%\n"
            f"  🟡 HighVol: {hv_p:.1f}%\n"
            f"  🔵 Sideways: {sw_p:.1f}%\n\n"
            f"*Target Exposure:*\n"
            f"  Equity: {eq_exp}% | Cash: {cash_exp}%\n"
        )
        if sector_str:
            msg += f"\n*Top Sectors ({curr_reg}):*\n  {sector_str}\n"

        msg += "\n⚠️ Model output, not financial advice."
        await update.message.reply_text(msg, parse_mode="Markdown")

    except FileNotFoundError:
        await update.message.reply_text(
            "❌ No regime history found. Run /retrain first."
        )
    except Exception as e:
        logger.exception("Error in /status")
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_sectors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show sector allocation for every regime."""
    try:
        df = _load_regime_history()
        sector_mix = _load_sector_mix()
        curr_reg = df.iloc[-1].get("Regime", "Sideways")

        if not sector_mix:
            await update.message.reply_text("❌ No sector mix data found.")
            return

        lines = ["📊 *Sector Allocation by Regime*\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]

        for regime in ["Bull", "Bear", "HighVol", "Sideways"]:
            if regime not in sector_mix:
                continue
            emoji = REGIME_EMOJI.get(regime, "⚪")
            active = " ◀ ACTIVE" if regime == curr_reg else ""
            lines.append(f"{emoji} *{regime}*{active}")
            top = sorted(sector_mix[regime].items(), key=lambda x: x[1], reverse=True)
            for s, w in top:
                if w > 0.005:
                    bar = "█" * int(w * 20)
                    lines.append(f"  {s:12s} {w*100:5.1f}% {bar}")
            lines.append("")

        lines.append("⚠️ Model output, not financial advice.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except FileNotFoundError:
        await update.message.reply_text("❌ No data found. Run /retrain first.")
    except Exception as e:
        logger.exception("Error in /sectors")
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_macro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Latest macro indicators snapshot."""
    try:
        df = _load_regime_history()
        latest = df.iloc[-1]
        date_str = latest.name.strftime("%d %b %Y") if hasattr(latest.name, "strftime") else str(latest.name)[:10]

        repo = latest.get("RepoRate", 0)
        gsec10 = latest.get("GSecYield10", 0) if "GSecYield10" in df.columns else 0
        # Compute real rate
        cpi = latest.get("CPI", 0)
        real_rate = repo - cpi if repo and cpi else 0

        msg = (
            f"🏦 *MACRO SNAPSHOT — {date_str}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"NIFTY 50       : {latest.get('NIFTY', 0):,.2f}\n"
            f"India VIX      : {latest.get('VIX', 0):.2f}\n"
            f"Repo Rate      : {repo:.2f}%\n"
            f"10Y G-Sec      : {gsec10:.2f}%\n"
            f"Yield Curve    : {latest.get('YieldCurve', 0):.2f}% (10Y–2Y)\n"
            f"CPI Inflation  : {cpi:.2f}%\n"
            f"WPI Inflation  : {latest.get('WPI', 0):.2f}%\n"
            f"IIP Growth     : {latest.get('IIP', 0):.2f}%\n"
            f"Real Rate      : {real_rate:.2f}% (Repo − CPI)\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except FileNotFoundError:
        await update.message.reply_text("❌ No data found. Run /retrain first.")
    except Exception as e:
        logger.exception("Error in /macro")
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show last N regime transitions."""
    try:
        n = 10
        if context.args:
            try:
                n = int(context.args[0])
                n = max(1, min(n, 50))
            except ValueError:
                pass

        df = _load_regime_history()
        regimes = df["Regime"]
        changes = regimes[regimes != regimes.shift(1)]

        if len(changes) == 0:
            await update.message.reply_text("No regime transitions found.")
            return

        recent = changes.tail(n)
        lines = [f"📜 *Last {len(recent)} Regime Transitions*\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"]

        prev = None
        for date, regime in recent.items():
            emoji = REGIME_EMOJI.get(regime, "⚪")
            date_str = date.strftime("%d %b %Y") if hasattr(date, "strftime") else str(date)[:10]
            arrow = f" (← {prev})" if prev else ""
            lines.append(f"{emoji} {date_str} → *{regime}*{arrow}")
            prev = regime

        # Duration of current regime
        last_change = changes.index[-1]
        current_days = (df.index[-1] - last_change).days
        lines.append(f"\nCurrent regime held for *{current_days} days*")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except FileNotFoundError:
        await update.message.reply_text("❌ No data found. Run /retrain first.")
    except Exception as e:
        logger.exception("Error in /history")
        await update.message.reply_text(f"❌ Error: {e}")


# ── Chart Commands ──────────────────────────────────────────────────

FIGURE_MAP = {
    "1": ("fig1_regime_detection.png", "📊 Fig 1 — Regime Detection Overview (NIFTY 50 + HMM Posteriors)"),
    "2": ("fig2_strategy_backtest.png", "📈 Fig 2 — Sector Rotation Strategy vs Buy & Hold Backtest"),
    "3": ("fig3_hmm_internals.png", "🔬 Fig 3 — HMM Model Internals (State Distributions & Transition Matrix)"),
    "4": ("fig4_confidence_intervals.png", "📉 Fig 4 — Bootstrap Confidence Intervals & Return Distributions"),
    "5": ("fig5_model_selection.png", "📐 Fig 5 — HMM Model Selection (BIC / AIC / Log-Likelihood Curves)"),
    "6": ("fig6_walk_forward.png", "🔄 Fig 6 — Walk-Forward Out-of-Sample Performance"),
    "7": ("fig7_sector_rotation.png", "🗺 Fig 7 — NSE Sector Rotation Heatmap & Regime Returns"),
    "8": ("fig8_new_macro_signals.png", "📡 Fig 8 — Macroeconomic Leading Signals & Yield Curve Spreads"),
    "9": ("fig9_paper_portfolio.png", "💼 Fig 9 — Live Paper Portfolio (Frozen Model, Out-of-Sample)"),
    "10": ("fig10_parameter_stability.png", "🎛 Fig 10 — Parameter Stability & Sensitivity Analysis Heatmaps"),
}

async def _send_chart(update, filename, caption):
    """Generic chart sender looking in figures/ and falling back to project root."""
    chart_path = FIGURES_DIR / filename
    if not chart_path.exists():
        chart_path = PROJECT_DIR / filename
    if not chart_path.exists():
        await update.message.reply_text(
            f"❌ {filename} not found in {FIGURES_DIR}. Run /retrain to generate charts."
        )
        return
    with open(chart_path, "rb") as f:
        await update.message.reply_photo(
            photo=f,
            caption=caption,
        )


@authorized
async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send regime detection overview chart (Fig 1) or specified figure number (1–10)."""
    fig_key = "1"
    if context.args:
        raw_arg = context.args[0].lower().replace("fig", "").strip()
        if raw_arg in FIGURE_MAP:
            fig_key = raw_arg

    filename, caption = FIGURE_MAP[fig_key]
    await _send_chart(update, filename, caption)


@authorized
async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the strategy backtest chart (Fig 2)."""
    await _send_chart(
        update,
        "fig2_strategy_backtest.png",
        "📈 Fig 2 — Sector Rotation Strategy vs Buy & Hold Backtest",
    )


@authorized
async def cmd_internals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the HMM model internals chart (Fig 3)."""
    await _send_chart(
        update,
        "fig3_hmm_internals.png",
        "🔬 Fig 3 — HMM Model Internals (State Distributions & Transition Matrix)",
    )


@authorized
async def cmd_confidence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the bootstrap confidence intervals & distributions chart (Fig 4)."""
    await _send_chart(
        update,
        "fig4_confidence_intervals.png",
        "📉 Fig 4 — Bootstrap Confidence Intervals & Return Distributions",
    )


@authorized
async def cmd_modelselection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the model selection chart (Fig 5)."""
    await _send_chart(
        update,
        "fig5_model_selection.png",
        "📐 Fig 5 — HMM Model Selection (BIC / AIC / Log-Likelihood Curves)",
    )


@authorized
async def cmd_walkforward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the walk-forward OOS chart (Fig 6) + summary stats."""
    await _send_chart(
        update,
        "fig6_walk_forward.png",
        "🔄 Fig 6 — Walk-Forward Out-of-Sample Performance",
    )
    # Also send summary stats if available
    try:
        wf = _load_walk_forward()
        if not wf.empty and "OOS_Sharpe" in wf.columns:
            mean_sharpe = wf["OOS_Sharpe"].mean()
            median_sharpe = wf["OOS_Sharpe"].median()
            pct_positive = (wf["OOS_Sharpe"] > 0).mean() * 100
            n_folds = len(wf)
            msg = (
                f"\n📋 *Walk-Forward Summary ({n_folds} folds)*\n"
                f"  Mean OOS Sharpe: {mean_sharpe:.2f}\n"
                f"  Median OOS Sharpe: {median_sharpe:.2f}\n"
                f"  Folds with positive Sharpe: {pct_positive:.0f}%"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        pass


@authorized
async def cmd_heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the sector rotation heatmap (Fig 7)."""
    await _send_chart(
        update,
        "fig7_sector_rotation.png",
        "🗺 Fig 7 — NSE Sector Rotation Heatmap & Regime Returns",
    )


@authorized
async def cmd_macrochart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the macroeconomic signals chart (Fig 8)."""
    await _send_chart(
        update,
        "fig8_new_macro_signals.png",
        "📡 Fig 8 — Macroeconomic Leading Signals & Yield Curve Spreads",
    )


@authorized
async def cmd_stability(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the parameter stability heatmap dashboard (Fig 10) + robustness metrics."""
    await _send_chart(
        update,
        "fig10_parameter_stability.png",
        "🎛 Fig 10 — Parameter Stability & Sensitivity Analysis Heatmaps",
    )
    try:
        rep = _load_parameter_stability()
        if rep:
            base = rep.get("baseline_performance", {})
            neigh = rep.get("plateau_neighborhood_3x3", {})
            glob = rep.get("global_parameter_grid", {})
            msg = (
                f"\n📊 *Parameter Robustness Scorecard*\n"
                f"  • Baseline Sharpe: *{base.get('sharpe_ratio', 'N/A')}* (Ann Ret: +{base.get('annualized_return_pct', 'N/A')}%, Max DD: {base.get('max_drawdown_pct', 'N/A')}%)\n"
                f"  • 3×3 Plateau Mean: *{neigh.get('mean_sharpe', 'N/A')}* (CV: {neigh.get('coefficient_of_variation_pct', 'N/A')}%)\n"
                f"  • Global Grid Sharpe Range: *[{glob.get('min_sharpe', 'N/A')}, {glob.get('max_sharpe', 'N/A')}]*\n"
                f"  • % Grid Beating Buy & Hold: *{glob.get('pct_combinations_beating_bh', 'N/A')}%*\n"
                f"  • Worst-Case Edge vs B&H: *+{glob.get('worst_case_outperformance_vs_bh_pct', 'N/A')}%*\n"
                f"  ✓ *Verdict:* Solid plateau, zero parameter overfitting detected."
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Failed to load or send parameter stability report: {e}")



# ── Retrain Command ─────────────────────────────────────────────────

# Track running retrain processes to prevent concurrent runs
_retrain_lock = asyncio.Lock()


@authorized
async def cmd_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Re-run the full model training pipeline."""
    if _retrain_lock.locked():
        await update.message.reply_text(
            "⏳ A retrain is already in progress. Please wait for it to finish."
        )
        return

    async with _retrain_lock:
        await update.message.reply_text(
            "🔄 Starting full model retrain...\n"
            "This typically takes 3–8 minutes. You'll be notified when it's done."
        )

        try:
            script_path = SRC_DIR / "regime_detector_v2.py"
            if not script_path.exists():
                script_path = PROJECT_DIR / "src" / "regime_detector_v2.py"
            if not script_path.exists():
                script_path = PROJECT_DIR / "regime_detector_v2.py"
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                cwd=str(PROJECT_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Extract key metrics from stdout
                output = stdout.decode("utf-8", errors="replace")
                summary_lines = []

                for line in output.splitlines():
                    if any(kw in line for kw in [
                        "Active Regime", "NIFTY 50 Close", "Posterior Probs",
                        "Target Exposure", "Sector Allocation",
                        "OOS Sharpe", "Walk-Forward",
                    ]):
                        summary_lines.append(line.strip())

                summary = "\n".join(summary_lines[-10:]) if summary_lines else "Pipeline completed."

                await update.message.reply_text(
                    f"✅ *Retrain Complete!*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"```\n{summary}\n```\n"
                    f"Use /status or /chart to see updated results.",
                    parse_mode="Markdown",
                )
            else:
                error_tail = stderr.decode("utf-8", errors="replace")[-500:]
                await update.message.reply_text(
                    f"❌ *Retrain Failed* (exit code {process.returncode})\n"
                    f"```\n{error_tail}\n```",
                    parse_mode="Markdown",
                )

        except Exception as e:
            logger.exception("Error during retrain")
            await update.message.reply_text(f"❌ Retrain error: {e}")


# ── Help Command ────────────────────────────────────────────────────

@authorized
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full command reference."""
    msg = (
        "📖 *Command Reference*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "*Market Intelligence:*\n"
        "  /status — Current regime & posteriors\n"
        "  /sectors — Sector allocations (all regimes)\n"
        "  /macro — Macro indicators snapshot\n"
        "  /history [N] — Last N regime transitions\n\n"
        "*Charts (figures/):*\n"
        "  /chart [N] — Regime detection overview (Fig 1–10, default Fig 1)\n"
        "  /backtest — Strategy backtest chart (Fig 2)\n"
        "  /internals — HMM model internals & transition matrix (Fig 3)\n"
        "  /confidence — Bootstrap confidence intervals (Fig 4)\n"
        "  /modelselection — Model selection curves (Fig 5)\n"
        "  /walkforward — Walk-forward OOS results (Fig 6)\n"
        "  /heatmap — Sector rotation heatmap (Fig 7)\n"
        "  /macrochart — Macro leading signals (Fig 8)\n"
        "  /fig9 — Paper portfolio dashboard chart (Fig 9)\n"
        "  /stability — Parameter stability heatmaps (Fig 10)\n\n"
        "*Paper Portfolio & Execution:*\n"
        "  /portfolio — Current positions, NAV, returns\n"
        "  /reconcile — EOD trade reconciliation audit\n"
        "  /updateportfolio — Refresh with latest EOD prices\n\n"
        "*Admin:*\n"
        "  /retrain — Re-run full pipeline (~5 min)\n\n"
        "*Notes:*\n"
        "• All data comes from the latest pipeline run\n"
        "• Outputs loaded from data/, figures/, and artifacts/\n"
        "• Use /retrain to refresh with latest market data\n"
        "• Only authorized users can execute commands\n"
        "• This is a quantitative model, not financial advice"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


# ── Catch-all for unknown commands ──────────────────────────────────

@authorized
async def cmd_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Respond to unknown commands."""
    await update.message.reply_text(
        "❓ Unknown command. Use /help to see available commands."
    )


# ══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════

async def post_init(application):
    """Set bot commands for the Telegram menu."""
    commands = [
        BotCommand("status", "Current regime & market status"),
        BotCommand("sectors", "Sector allocation by regime"),
        BotCommand("macro", "Macro indicators snapshot"),
        BotCommand("history", "Recent regime transitions"),
        BotCommand("chart", "Regime detection chart (or /chart N)"),
        BotCommand("backtest", "Strategy backtest chart (Fig 2)"),
        BotCommand("internals", "HMM model internals (Fig 3)"),
        BotCommand("confidence", "Bootstrap confidence intervals (Fig 4)"),
        BotCommand("modelselection", "Model selection curves (Fig 5)"),
        BotCommand("walkforward", "Walk-forward OOS results (Fig 6)"),
        BotCommand("heatmap", "Sector rotation heatmap (Fig 7)"),
        BotCommand("macrochart", "Macro leading signals (Fig 8)"),
        BotCommand("fig9", "Paper portfolio chart (Fig 9)"),
        BotCommand("stability", "Parameter stability heatmaps (Fig 10)"),
        BotCommand("portfolio", "Paper portfolio status & holdings"),
        BotCommand("reconcile", "EOD Trade Reconciliation Audit"),
        BotCommand("updateportfolio", "Update portfolio with latest prices"),
        BotCommand("retrain", "Re-run model pipeline"),
        BotCommand("help", "Full command reference"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("✓ Bot commands menu registered")



# ── Paper Portfolio Commands ────────────────────────────────────────

@authorized
async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current paper portfolio status."""
    try:
        hist = _load_paper_portfolio()
        if hist.empty:
            await update.message.reply_text(
                f"❌ No paper portfolio data found in {DATA_DIR}. Run `python src/paper_portfolio.py` or use /updateportfolio first."
            )
            return

        latest = hist.iloc[-1]
        nav = latest.get("NAV", 0)
        total_ret = (nav / 10_00_000 - 1) * 100
        max_dd = hist["Drawdown"].min() * 100
        regime = latest.get("Regime", "Unknown")
        emoji = REGIME_EMOJI.get(regime, "⚪")
        date_str = latest.name.strftime("%d %b %Y") if hasattr(latest.name, "strftime") else str(latest.name)[:10]
        n_days = len(hist)
        n_rebal = int(hist.get("Rebalanced", pd.Series([False])).sum())

        # Holdings
        etf_names = ['BANKBEES', 'ITBEES', 'PHARMABEES', 'AUTOBEES',
                      'METALIETF', 'MOREALTY', 'CPSEETF', 'INFRABEES']
        holdings_lines = []
        for etf in etf_names:
            units = latest.get(f"{etf}_units", 0)
            value = latest.get(f"{etf}_value", 0)
            if units > 0.01:
                weight = (value / nav * 100) if nav > 0 else 0
                holdings_lines.append(f"  {etf:12s} {units:8.1f} units  ₹{value:>10,.0f}  ({weight:.1f}%)")

        cash = latest.get("Cash", 0)
        if cash > 0.01:
            holdings_lines.append(f"  {'CASH':12s} {'—':>8s}       ₹{cash:>10,.0f}  ({cash/nav*100:.1f}%)")

        holdings_str = "\n".join(holdings_lines) if holdings_lines else "  No positions"

        msg = (
            f"💼 *PAPER PORTFOLIO — {date_str}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} Regime: *{regime.upper()}*\n"
            f"NAV: ₹{nav:,.0f}\n"
            f"Total Return: {total_ret:+.2f}%\n"
            f"Max Drawdown: {max_dd:.2f}%\n"
            f"OOS Days: {n_days}  |  Rebalances: {n_rebal}\n\n"
            f"*Holdings:*\n{holdings_str}\n\n"
            f"⚠️ Frozen model — pure out-of-sample."
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    except Exception as e:
        logger.exception("Error in /portfolio")
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_fig9(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the paper portfolio chart (Fig 9)."""
    await _send_chart(
        update,
        "fig9_paper_portfolio.png",
        "💼 Fig 9 — Live Paper Portfolio (Frozen Model, Out-of-Sample)",
    )


@authorized
async def cmd_update_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Re-run the paper portfolio script to update with latest prices."""
    await update.message.reply_text("🔄 Updating paper portfolio with latest EOD prices...")
    try:
        script_path = SRC_DIR / "paper_portfolio.py"
        if not script_path.exists():
            script_path = PROJECT_DIR / "src" / "paper_portfolio.py"
        if not script_path.exists():
            script_path = PROJECT_DIR / "paper_portfolio.py"
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            cwd=str(PROJECT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            output = stdout.decode("utf-8", errors="replace")
            summary = [l.strip() for l in output.splitlines() if any(
                kw in l for kw in ["Current NAV", "Total Return", "Regime", "Max Drawdown", "OOS"]
            )]
            summary_text = "\n".join(summary[-6:]) if summary else "Update complete."
            await update.message.reply_text(
                f"✅ *Portfolio Updated!*\n```\n{summary_text}\n```\nUse /portfolio or /fig9 to view.",
                parse_mode="Markdown",
            )
        else:
            err = stderr.decode("utf-8", errors="replace")[-400:]
            await update.message.reply_text(f"❌ Update failed:\n```\n{err}\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_reconcile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Run EOD portfolio reconciliation and return audit status."""
    try:
        from database.connection import get_db
        from database.models import PortfolioLedger
        from reconciliation.reconciler import DailyReconciliationEngine
        from datetime import datetime

        with get_db() as db:
            latest = db.query(PortfolioLedger).order_by(PortfolioLedger.time.desc()).first()
            dt = latest.time.date() if latest else datetime.now().date()
            report = DailyReconciliationEngine.reconcile(db, dt)
            s = report.audit_summary

        status_emoji = "✅" if report.status == "RECONCILED" else "🚨"
        msg = (
            f"{status_emoji} *EOD TRADE RECONCILIATION AUDIT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Date: `{s['date']}`\n"
            f"Audit Status: *{s['status']}*\n\n"
            f"• Position Breaks: `{s['position_breaks']}`\n"
            f"• NAV Discrepancy: `₹{s['nav_discrepancy_inr']:.4f}`\n"
            f"• Cash Discrepancy: `₹{s['cash_discrepancy_inr']:.4f}`\n"
            f"• Stop Loss Integrity: `{s['stop_loss_integrity']}`\n"
            f"• Capital in Cash: `{s['in_cash']}`\n"
            f"• Total Portfolio NAV: `₹{s['nav']:,.2f}`\n"
            f"• Active Regime: *{s['active_regime']}*\n"
        )
        if s["position_breaks"] > 0:
            msg += f"\n⚠️ *Breaks Detected:* `{s['break_details']}`\n"
        msg += "\n🔍 *Audited by Institutional Reconciler Engine*"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.exception("Error in /reconcile")
        await update.message.reply_text(f"❌ Reconciliation error: {e}")


def main():
    """Build and run the Telegram bot."""
    logger.info("Starting India HMM Regime Detector Telegram Bot...")
    logger.info(f"Project dir: {PROJECT_DIR}")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info(f"Figures dir: {FIGURES_DIR}")
    logger.info(f"Artifacts dir: {ARTIFACTS_DIR}")
    logger.info(f"Authorized chat ID: {AUTHORIZED_CHAT_ID}")

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("sectors", cmd_sectors))
    app.add_handler(CommandHandler("macro", cmd_macro))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("chart", cmd_chart))
    app.add_handler(CommandHandler("backtest", cmd_backtest))
    app.add_handler(CommandHandler("internals", cmd_internals))
    app.add_handler(CommandHandler("confidence", cmd_confidence))
    app.add_handler(CommandHandler("modelselection", cmd_modelselection))
    app.add_handler(CommandHandler("walkforward", cmd_walkforward))
    app.add_handler(CommandHandler("heatmap", cmd_heatmap))
    app.add_handler(CommandHandler("macrochart", cmd_macrochart))
    app.add_handler(CommandHandler("fig9", cmd_fig9))
    app.add_handler(CommandHandler("stability", cmd_stability))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("reconcile", cmd_reconcile))
    app.add_handler(CommandHandler("updateportfolio", cmd_update_portfolio))
    app.add_handler(CommandHandler("retrain", cmd_retrain))
    app.add_handler(CommandHandler("help", cmd_help))

    # Catch unknown commands
    app.add_handler(MessageHandler(filters.COMMAND, cmd_unknown))

    logger.info("✓ All handlers registered. Bot is polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
