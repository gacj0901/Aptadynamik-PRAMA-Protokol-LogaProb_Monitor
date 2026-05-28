from pathlib import Path

path = Path("src/aptadynamik/pipelines/v2.py")
text = path.read_text(encoding="utf-8").replace("\r\n", "\n")

def replace_once(src, old, new):
    if old not in src:
        raise SystemExit(f"Pattern not found:\n{old[:300]}")
    return src.replace(old, new, 1)

if "def variance(values, mean):" not in text:
    text = replace_once(
        text,
        "\ndef window_aggregate(signals, window_size=WINDOW_SIZE):\n",
        """
def variance(values, mean):
    if not values:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def window_aggregate(signals, window_size=WINDOW_SIZE):
"""
    )

text = replace_once(
    text,
    """        avg_gap = sum(s['gap'] for s in chunk) / len(chunk)
        avg_entropy = sum(s['entropy'] for s in chunk) / len(chunk)
        avg_logprob = sum(s['top1_logprob'] for s in chunk) / len(chunk)
""",
    """        gap_values = [s['gap'] for s in chunk]
        entropy_values = [s['entropy'] for s in chunk]
        logprob_values = [s['top1_logprob'] for s in chunk]

        avg_gap = sum(gap_values) / len(gap_values)
        avg_entropy = sum(entropy_values) / len(entropy_values)
        avg_logprob = sum(logprob_values) / len(logprob_values)

        entropy_var = variance(entropy_values, avg_entropy)
        entropy_std = math.sqrt(entropy_var)
        entropy_range = max(entropy_values) - min(entropy_values)

        gap_var = variance(gap_values, avg_gap)
        gap_std = math.sqrt(gap_var)
"""
)

text = replace_once(
    text,
    """            'avg_gap': round(avg_gap, 4),
            'avg_entropy': round(avg_entropy, 4),
            'avg_logprob': round(avg_logprob, 4),
            **derived,
""",
    """            'avg_gap': round(avg_gap, 4),
            'avg_entropy': round(avg_entropy, 4),
            'entropy_var': round(entropy_var, 4),
            'entropy_std': round(entropy_std, 4),
            'entropy_range': round(entropy_range, 4),
            'gap_std': round(gap_std, 4),
            'avg_logprob': round(avg_logprob, 4),
            **derived,
"""
)

text = replace_once(
    text,
    """            'avg_gap': w['avg_gap'],
            'avg_entropy': w['avg_entropy'],
            'gap_norm': w['gap_norm'],
""",
    """            'avg_gap': w['avg_gap'],
            'avg_entropy': w['avg_entropy'],
            'entropy_var': w['entropy_var'],
            'entropy_std': w['entropy_std'],
            'entropy_range': w['entropy_range'],
            'gap_std': w['gap_std'],
            'gap_norm': w['gap_norm'],
"""
)

text = replace_once(
    text,
    """        'avg_margin': round(sum(s['margin'] for s in steps) / n_w, 4),
        'avg_entropy': round(sum(s['avg_entropy'] for s in steps) / n_w, 4),
""",
    """        'avg_margin': round(sum(s['margin'] for s in steps) / n_w, 4),
        'avg_entropy': round(sum(s['avg_entropy'] for s in steps) / n_w, 4),
        'avg_entropy_std': round(sum(s['entropy_std'] for s in steps) / n_w, 4),
        'max_entropy_std': round(max(s['entropy_std'] for s in steps), 4),
        'avg_entropy_range': round(sum(s['entropy_range'] for s in steps) / n_w, 4),
"""
)

text = replace_once(
    text,
    """                          f"mar={result['avg_margin']:.3f} | "
                          f"xi/w={result['xi_per_window']:.3f} "
""",
    """                          f"mar={result['avg_margin']:.3f} "
                          f"estd={result['avg_entropy_std']:.3f} | "
                          f"xi/w={result['xi_per_window']:.3f} "
"""
)

text = replace_once(
    text,
    """        ("Entropy (raw)", "avg_entropy"),
    ]
""",
    """        ("Entropy (raw)", "avg_entropy"),
        ("Entropy std", "avg_entropy_std"),
        ("Max ent std", "max_entropy_std"),
        ("Entropy range", "avg_entropy_range"),
    ]
"""
)

text = replace_once(
    text,
    """                  'final_xi', 'final_regime', 'avg_entropy']
""",
    """                  'final_xi', 'final_regime', 'avg_entropy',
                  'avg_entropy_std', 'max_entropy_std', 'avg_entropy_range']
"""
)

text = replace_once(
    text,
    """                  'avg_gap', 'avg_entropy', 'gap_norm', 'entropy_norm',
                  'rigidity', 'uncertainty', 'margin',
""",
    """                  'avg_gap', 'avg_entropy', 'entropy_var', 'entropy_std',
                  'entropy_range', 'gap_std', 'gap_norm', 'entropy_norm',
                  'rigidity', 'uncertainty', 'margin',
"""
)

path.write_text(text, encoding="utf-8")
print("Updated src/aptadynamik/pipelines/v2.py with intra-window entropy variance metrics.")