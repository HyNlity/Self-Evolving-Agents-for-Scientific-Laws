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

# Critic runtime artifact names
CRITIC_REPORT_MD = "critic_report.md"
CRITIC_REPORT_JSON = "critic_report.json"

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
