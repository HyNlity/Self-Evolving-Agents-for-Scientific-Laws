"""Hamilton playground constants."""

# plan.md Current Best block markers
CURRENT_BEST_BEGIN = "<!-- EVO_CURRENT_BEST_BEGIN -->"
CURRENT_BEST_END = "<!-- EVO_CURRENT_BEST_END -->"

# plan.md Strategy Queue block markers
STRATEGY_QUEUE_BEGIN = "<!-- EVO_STRATEGY_QUEUE_BEGIN -->"
STRATEGY_QUEUE_END = "<!-- EVO_STRATEGY_QUEUE_END -->"

# L3 experience.md block markers
EXPERIENCE_POSITIVE_BEGIN = "<!-- EVO_EXPERIENCE_POSITIVE_BEGIN -->"
EXPERIENCE_POSITIVE_END = "<!-- EVO_EXPERIENCE_POSITIVE_END -->"
EXPERIENCE_NEGATIVE_BEGIN = "<!-- EVO_EXPERIENCE_NEGATIVE_BEGIN -->"
EXPERIENCE_NEGATIVE_END = "<!-- EVO_EXPERIENCE_NEGATIVE_END -->"

# Critic trigger defaults
CRITIC_ROUND_INTERVAL = 5          # minimum rounds between critic interventions
CRITIC_MSE_PLATEAU_THRESHOLD = 0.01  # <1% improvement triggers critic
CRITIC_MSE_PLATEAU_WINDOW = 3      # consecutive rounds to check for plateau

# findings.md append marker
FINDINGS_APPEND = "<!-- EVO_FINDINGS_APPEND -->"
