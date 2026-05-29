import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ONSET_TOPIC_LIMIT = int(os.getenv("ONSET_TOPIC_LIMIT", "30"))
ONSET_MAX_TOKENS_STAGE1 = int(os.getenv("ONSET_MAX_TOKENS_STAGE1", "400"))
ONSET_MAX_TOKENS_STAGE2 = int(os.getenv("ONSET_MAX_TOKENS_STAGE2", "400"))
ONSET_WINDOW_SIZE = int(os.getenv("ONSET_WINDOW_SIZE", "32"))
ONSET_BETAS = [float(x.strip()) for x in os.getenv("ONSET_BETAS", "0.3,0.6,0.9").split(",") if x.strip()]
TOP_LOGPROBS = int(os.getenv("ONSET_TOP_LOGPROBS", "5"))

PSI_LOW = 0.2
PSI_CONTRADICTORY = 1.2
PSI_SATURATION = 1.8
S0 = 1.0

CONDITIONS = ("control", "contradictory_onset", "saturation_onset")

TOPICS = [
    "history of bridges",
    "how forests regenerate",
    "development of calendars",
    "ocean currents",
    "ceramics and pottery",
    "public libraries",
    "migration of birds",
    "evolution of writing systems",
    "architecture of train stations",
    "history of textile dyes",
    "how rivers shape valleys",
    "development of musical notation",
    "urban parks and civic life",
    "history of maps",
    "traditional bread making",
    "formation of coral reefs",
    "evolution of clocks",
    "botanical gardens",
    "history of glassmaking",
    "maintenance of hiking trails",
    "development of postal systems",
    "water mills in rural economies",
    "history of astronomy instruments",
    "how wetlands filter water",
    "origins of paper",
    "craft of bookbinding",
    "development of lighthouses",
    "history of marketplaces",
    "soil formation",
    "architecture of courtyards",
    "evolution of bicycles",
    "history of weather forecasting",
    "how glaciers move",
    "traditional fishing boats",
    "development of public museums",
    "history of spices",
    "irrigation canals",
    "evolution of alphabets",
    "history of gardens",
    "how bees communicate",
    "development of libraries in monasteries",
    "architecture of observatories",
    "history of ceramics kilns",
    "formation of deltas",
    "craft of weaving",
    "evolution of musical instruments",
    "history of road signs",
    "how seeds disperse",
    "development of harbor towns",
    "history of ink",
    "architecture of town halls",
    "how caves form",
    "development of cooking vessels",
    "history of public fountains",
    "migration routes in human history",
    "evolution of farm tools",
    "history of paper money",
    "how dunes move",
    "development of botanical classification",
    "architecture of covered markets",
]


def entropy_from_top_logprobs(top_logprobs):
    if not top_logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    norm = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-15) for p in norm)


def extract_openai_signals(logprobs_content):
    signals = []
    for token_data in logprobs_content or []:
        top_items = getattr(token_data, "top_logprobs", None) or []
        top_lps = [item.logprob for item in top_items if hasattr(item, "logprob")]
        signals.append(
            {
                "token": getattr(token_data, "token", ""),
                "top1_logprob": float(getattr(token_data, "logprob", 0.0)),
                "entropy": float(entropy_from_top_logprobs(top_lps)),
            }
        )
    return signals


def synthetic_signals(n_tokens, phase, condition):
    base = {
        ("pre", "control"): 0.45,
        ("post", "control"): 0.46,
        ("pre", "contradictory_onset"): 0.45,
        ("post", "contradictory_onset"): 0.92,
        ("pre", "saturation_onset"): 0.45,
        ("post", "saturation_onset"): 0.72,
    }[(phase, condition)]
    signals = []
    max_entropy = math.log2(TOP_LOGPROBS)
    for i in range(n_tokens):
        wiggle = 0.04 * math.sin(i / 7.0)
        entropy = max(0.0, min(max_entropy, (base + wiggle) * max_entropy))
        signals.append({"token": f"t{i}", "top1_logprob": -0.2, "entropy": entropy})
    return signals


def normalized_entropy(entropy_raw):
    max_entropy = math.log2(TOP_LOGPROBS)
    return min(1.0, max(0.0, entropy_raw / max(max_entropy, 1e-12)))


def psi_for(condition, phase):
    if phase == "pre" or condition == "control":
        return PSI_LOW
    if condition == "contradictory_onset":
        return PSI_CONTRADICTORY
    if condition == "saturation_onset":
        return PSI_SATURATION
    raise ValueError(f"Unknown condition: {condition}")


def window_phase_signals(signals, topic_id, condition, phase, start_index=0, window_size=ONSET_WINDOW_SIZE):
    rows = []
    for offset in range(0, len(signals), window_size):
        chunk = signals[offset : offset + window_size]
        if not chunk:
            continue
        entropy_raw = sum(item["entropy"] for item in chunk) / len(chunk)
        entropy_norm = normalized_entropy(entropy_raw)
        psi = psi_for(condition, phase)
        rows.append(
            {
                "topic_id": topic_id,
                "condition": condition,
                "phase": phase,
                "window_index": start_index + len(rows),
                "entropy_raw": round(entropy_raw, 6),
                "entropy_norm": round(entropy_norm, 6),
                "psi": psi,
                "delta": round(abs(entropy_norm - psi), 6),
                "n_tokens_in_window": len(chunk),
            }
        )
    return rows


def reconstruct_xi_lm(deltas, beta, s0=S0):
    xi = []
    for t in range(len(deltas)):
        acc = 0.0
        for tau in range(t + 1):
            acc += ((t - tau) + s0) ** (-beta) * deltas[tau]
        xi.append(acc)
    return xi


def add_xi_columns(windows, betas):
    deltas = [row["delta"] for row in windows]
    for beta in betas:
        xi_values = reconstruct_xi_lm(deltas, beta)
        key = beta_key(beta)
        for row, xi in zip(windows, xi_values):
            row[f"xi_lm_beta_{key}"] = round(xi, 6)
    return windows


def slope(values):
    if len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denom


def beta_key(beta):
    return str(beta).replace(".", "_")


def summarize_trajectory(topic_id, topic, condition, windows, betas):
    summary = {
        "topic_id": topic_id,
        "topic": topic,
        "condition": condition,
        "post_token_count": sum(row["n_tokens_in_window"] for row in windows if row["phase"] == "post"),
    }
    summary["completed_post"] = summary["post_token_count"] >= 0.8 * ONSET_MAX_TOKENS_STAGE2

    for beta in betas:
        key = beta_key(beta)
        pre = [row[f"xi_lm_beta_{key}"] for row in windows if row["phase"] == "pre"]
        post = [row[f"xi_lm_beta_{key}"] for row in windows if row["phase"] == "post"]
        pre_slope = slope(pre)
        post_slope = slope(post)
        summary[f"slope_pre_xi_beta_{key}"] = round(pre_slope, 6)
        summary[f"slope_post_xi_beta_{key}"] = round(post_slope, 6)
        summary[f"post_minus_pre_slope_beta_{key}"] = round(post_slope - pre_slope, 6)

    return summary


def difference_in_differences(treatment_summary, control_summary, beta):
    key = beta_key(beta)
    return (
        treatment_summary[f"post_minus_pre_slope_beta_{key}"]
        - control_summary[f"post_minus_pre_slope_beta_{key}"]
    )


def survival_ratio(saturation_post_tokens, control_post_tokens):
    if control_post_tokens <= 0:
        return 0.0
    return saturation_post_tokens / control_post_tokens


def stage1_prompt(topic):
    return (
        f"Write a neutral, continuous explanatory passage about {topic}. "
        "Avoid argument, contradiction, lists, and restrictive formatting."
    )


def stage2_prompt(condition, topic):
    if condition == "control":
        return f"Continue the same neutral explanatory passage about {topic} in the same style."
    if condition == "contradictory_onset":
        return (
            f"Continue the passage about {topic}, but now sustain an incompatible constraint: "
            "be highly detailed while keeping every sentence extremely brief, remain completely neutral "
            "while making a persuasive argument, and add new information while avoiding any new claims."
        )
    if condition == "saturation_onset":
        return (
            f"Continue the passage about {topic}, but obey many simultaneous constraints: each sentence "
            "must be under twelve words, no word may repeat, include a date, a place, a material, and a "
            "process in every sentence, avoid adjectives, use alternating sentence lengths, and keep the "
            "tone neutral."
        )
    raise ValueError(f"Unknown condition: {condition}")


def make_openai_call():
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def call(messages, max_tokens):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        content = choice.logprobs.content if choice.logprobs else []
        return text, extract_openai_signals(content)

    return call


def make_dry_call():
    def call(messages, max_tokens):
        text = " ".join(f"token{i}" for i in range(max_tokens))
        last = messages[-1]["content"].lower()
        if "many simultaneous constraints" in last:
            condition = "saturation_onset"
            phase = "post"
            n_tokens = max(1, int(max_tokens * 0.65))
        elif "incompatible constraint" in last:
            condition = "contradictory_onset"
            phase = "post"
            n_tokens = max_tokens
        elif "continue" in last:
            condition = "control"
            phase = "post"
            n_tokens = max_tokens
        else:
            condition = "control"
            phase = "pre"
            n_tokens = max_tokens
        return text, synthetic_signals(n_tokens, phase, condition)

    return call


def run_trajectory(topic_id, topic, condition, call_fn, betas):
    stage1_messages = [
        {"role": "system", "content": "Write clear neutral expository prose."},
        {"role": "user", "content": stage1_prompt(topic)},
    ]
    prefix_text, pre_signals = call_fn(stage1_messages, ONSET_MAX_TOKENS_STAGE1)

    stage2_messages = [
        {"role": "system", "content": "Write clear neutral expository prose unless the user changes constraints."},
        {"role": "user", "content": stage1_prompt(topic)},
        {"role": "assistant", "content": prefix_text},
        {"role": "user", "content": stage2_prompt(condition, topic)},
    ]
    continuation_text, post_signals = call_fn(stage2_messages, ONSET_MAX_TOKENS_STAGE2)

    pre_windows = window_phase_signals(pre_signals, topic_id, condition, "pre", start_index=0)
    post_windows = window_phase_signals(
        post_signals, topic_id, condition, "post", start_index=len(pre_windows)
    )
    windows = add_xi_columns(pre_windows + post_windows, betas)
    summary = summarize_trajectory(topic_id, topic, condition, windows, betas)

    return {
        "topic_id": topic_id,
        "topic": topic,
        "condition": condition,
        "prefix_preview": prefix_text[:300],
        "continuation_preview": continuation_text[:300],
        "summary": summary,
        "windows": windows,
    }


def analyze_summaries(summary_rows, betas):
    by_topic_condition = {
        (row["topic_id"], row["condition"]): row
        for row in summary_rows
    }
    topic_ids = sorted({row["topic_id"] for row in summary_rows})
    did_contradictory = {beta: [] for beta in betas}
    did_saturation = {beta: [] for beta in betas}
    survival_values = []

    for topic_id in topic_ids:
        control = by_topic_condition.get((topic_id, "control"))
        contradictory = by_topic_condition.get((topic_id, "contradictory_onset"))
        saturation = by_topic_condition.get((topic_id, "saturation_onset"))
        if not control or not contradictory or not saturation:
            continue
        for beta in betas:
            did_contradictory[beta].append(difference_in_differences(contradictory, control, beta))
            did_saturation[beta].append(difference_in_differences(saturation, control, beta))
        survival_values.append(
            survival_ratio(saturation["post_token_count"], control["post_token_count"])
        )

    return {
        "topic_ids": topic_ids,
        "did_contradictory": did_contradictory,
        "did_saturation": did_saturation,
        "survival_saturation": survival_values,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_report(summary_rows, betas):
    analysis = analyze_summaries(summary_rows, betas)
    topic_count = len(analysis["topic_ids"])
    trajectory_count = len(summary_rows)

    print("=" * 78)
    print("PRAMA onset experiment")
    print("=" * 78)
    print(f"Topics: {topic_count}")
    print(f"Trajectories: {trajectory_count}")

    contradiction_passes = []
    for beta in betas:
        values = analysis["did_contradictory"][beta]
        positive_count = sum(1 for value in values if value > 0)
        proportion = positive_count / len(values) if values else 0.0
        contradiction_passes.append(mean(values) > 0 and proportion >= 0.6)
        print(
            f"Mean DiD_contradictory beta {beta}: {mean(values):.6f} | "
            f"topics > 0: {positive_count}/{len(values)}"
        )

    survival = analysis["survival_saturation"]
    saturation_lower_count = sum(1 for value in survival if value < 1.0)
    saturation_lower_prop = saturation_lower_count / len(survival) if survival else 0.0
    saturation_pass = mean(survival) < 0.90 and saturation_lower_prop >= 0.6
    print(f"Mean survival_saturation: {mean(survival):.6f}")
    print(f"Topics where saturation survival < control survival: {saturation_lower_count}/{len(survival)}")

    contradiction_pass = all(contradiction_passes) if contradiction_passes else False
    print("Interpretation:")
    if contradiction_pass and saturation_pass:
        print("  PASS: contradiction increases post-onset Xi slope and saturation reduces post-onset survival.")
    elif contradiction_pass:
        print("  PARTIAL: contradiction endpoint passes; saturation survival endpoint does not pass.")
    elif saturation_pass:
        print("  PARTIAL: saturation survival endpoint passes; contradiction slope endpoint does not pass.")
    else:
        print("  FAIL: preregistered onset endpoints did not pass in this run.")


def write_outputs(results, summary_rows, detail_rows, betas):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"onset_results_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_path = results_dir / f"onset_summary_{timestamp}.csv"
    summary_fields = ["topic_id", "topic", "condition", "post_token_count", "completed_post"]
    for beta in betas:
        key = beta_key(beta)
        summary_fields.extend(
            [
                f"slope_pre_xi_beta_{key}",
                f"slope_post_xi_beta_{key}",
                f"post_minus_pre_slope_beta_{key}",
            ]
        )
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in summary_fields})

    detail_path = results_dir / f"onset_detail_{timestamp}.csv"
    detail_fields = [
        "topic_id",
        "condition",
        "phase",
        "window_index",
        "entropy_raw",
        "entropy_norm",
        "psi",
        "delta",
    ]
    detail_fields.extend(f"xi_lm_beta_{beta_key(beta)}" for beta in betas)
    detail_fields.append("n_tokens_in_window")
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({field: row.get(field) for field in detail_fields})

    print(f"\n-> {json_path}")
    print(f"-> {summary_path}")
    print(f"-> {detail_path}")


def main():
    topic_limit = int(os.getenv("ONSET_TOPIC_LIMIT", str(ONSET_TOPIC_LIMIT)))
    topics = TOPICS[:topic_limit]
    dry_run = not os.environ.get("OPENAI_API_KEY")
    call_fn = make_dry_call() if dry_run else make_openai_call()

    if dry_run:
        print("OPENAI_API_KEY not set; running deterministic dry run backend.")

    results = []
    summary_rows = []
    detail_rows = []

    for topic_id, topic in enumerate(topics, start=1):
        print(f"\nTopic {topic_id}/{len(topics)}: {topic}")
        for condition in CONDITIONS:
            print(f"  {condition}")
            result = run_trajectory(topic_id, topic, condition, call_fn, ONSET_BETAS)
            results.append(result)
            summary_rows.append(result["summary"])
            detail_rows.extend(result["windows"])

    print_report(summary_rows, ONSET_BETAS)
    write_outputs(results, summary_rows, detail_rows, ONSET_BETAS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
