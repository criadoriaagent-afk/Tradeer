"""
Tradeer Package Initialization.
Exposes core, engines, audits, and research modules for complete backward compatibility.
"""

from .core import (
    config,
    data_loader,
    live_feeder,
    paper_trading,
    bybit_connector,
    risk_manager,
    strategy,
    validation_matrix,
    execution_engine,
    notifier,
    telegram_bot,
    price_action,
    probability_calculator,
    sentiment_engine,
)

from .engines import (
    backtester,
    portfolio_backtester,
    adaptive_ai,
    cross_asset,
    institutional_engine,
    monte_carlo,
    optimizer,
    predictive_engine,
    self_healing,
    forward_oos_cloud_engine,
    forward_oos_futures_cloud_engine,
    spot_signal_futures_cloud_engine,
)

from .audits import (
    early_prune_freeze_audit,
    early_prune_rigorous_validator,
    entry_identity_auditor,
    entry_quality_investigator,
    er_hypothesis_falsification_suite,
    final_integrity_and_equity_suite,
    methodology_audit_suite,
    post_entry_divergence_investigator,
    prospective_paper_trading_suite,
    robustness_falsifier,
    trade_auditor,
    v1_audit_suite,
    v1_autopsy_diagnostics,
    v1_bottleneck_investigator,
    v1_canonical_engine,
    v1_exit_research_suite,
    v1_final_integrity_suite,
    v1_integrity_audit_suite,
    v1_simplified_1d_engine,
)

from .research import (
    ablation_study,
    atr_sweep_test,
    early_prune_futures_research,
    regime_exit_research,
    v2_cross_asset_diagnostic,
    v2_oos_validation_suite,
    v2_trend_reversal_engine,
)
