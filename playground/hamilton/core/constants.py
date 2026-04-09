"""Hamilton playground constants."""

# plan.md Current Best block markers
CURRENT_BEST_BEGIN = "<!-- EVO_CURRENT_BEST_BEGIN -->"
CURRENT_BEST_END = "<!-- EVO_CURRENT_BEST_END -->"

# plan.md Strategy Queue block markers
STRATEGY_QUEUE_BEGIN = "<!-- EVO_STRATEGY_QUEUE_BEGIN -->"
STRATEGY_QUEUE_END = "<!-- EVO_STRATEGY_QUEUE_END -->"

# L3 runtime artifact names
TASK_SIGNATURE_FILE = "task_signature.json"
L3_HITS_FILE = "l3_hits.json"
L3_CONTEXT_FILE = "l3_context.md"
CRITIC_CONTEXT_FILE = "critic_context.md"
DEBATE_STATE_FILE = "debate_state.json"
ENV_CAPABILITIES_FILE = "environment_capabilities.json"
EVALUATION_CONTEXT_FILE = "evaluation_context.json"
EVALUATION_CONTEXT_MD = "evaluation_context.md"
CRITIC_SCHEDULER_STATE_FILE = "critic_scheduler_state.json"
HCC_LEDGER_FILE = "hcc_ledger.jsonl"

# Critic runtime artifact names
CRITIC_REPORT_MD = "critic_report.md"
CRITIC_REPORT_JSON = "critic_report.json"
CRITIC_ATTACK_PLAN_JSON = "critic_attack_plan.json"
CRITIC_ATTACK_LOG_JSONL = "critic_attack_log.jsonl"
CRITIC_ATTACKS_DIR = "critic_attacks"

# L3 store layout
L3_INDEX_FILE = "index.jsonl"
L3_TASKS_DIR = "tasks"

# Challenge taxonomy used by the critic.
CRITIC_CHALLENGE_TYPES = (
    "support_set_attack",
    "structure_attack",
    "ood_generalization_attack",
    "physics_consistency_attack",
    "numerical_stability_attack",
    "evidence_gap_attack",
)

CRITIC_INTERVENTION_TYPES = (
    "extreme_value_probe",
    "support_ablation",
    "ood_slice_probe",
    "perturbation_probe",
    "physics_counterexample",
    "short_ivp_integration",
    "result_file_check",
)
