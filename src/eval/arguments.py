ARGUMENTS = {
    "task": "Sentiment, market-direction, or agent-return task to evaluate.",
    "system": "News signal version or agent system to evaluate.",
    "dataset": "Optional path overriding the task's default dataset.",
    "model": "Optional model overriding the selected system default.",
    "output": "Optional path overriding the task and system report filename.",
    "limit": "Example limit, defaulting to eight review-priority agent examples.",
    "all": "Evaluate every agent example instead of the default limited selection.",
}

SEED_ARGUMENTS = {
    "seed": "Path to the recent headline collection manifest.",
    "seed_output": "Path for the generated headline-level JSONL dataset.",
    "execute": "Perform external requests instead of a dry run.",
    "overwrite": "Replace an existing generated dataset.",
}
