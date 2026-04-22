import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CCR_STATS_RE = re.compile(
    r"client\s+(?P<client>\d+)\s+round\s+(?P<round>\d+)\s+complementary routing stats:"
    r"\s+confidence=(?P<confidence>[-+0-9.eE]+)"
    r"\s+need=(?P<need>[-+0-9.eE]+)"
    r"\s+slots=(?P<slots>\[[^\]]+\])"
)


FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze v3.1 CCR client logs for confidence-slot structure."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to logs/run_xxx directory containing client*.log",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for CSV/JSON outputs. Defaults to <log_dir>/ccr_analysis",
    )
    parser.add_argument(
        "--flat_threshold",
        type=float,
        default=0.02,
        help="Heuristic std threshold used for interpretation.",
    )
    return parser.parse_args()


def parse_slot_array(slot_text):
    values = [float(x) for x in FLOAT_RE.findall(slot_text)]
    return values


def parse_client_log(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = CCR_STATS_RE.search(line)
        if not match:
            continue
        slot_values = parse_slot_array(match.group("slots"))
        rows.append(
            {
                "client": int(match.group("client")),
                "round": int(match.group("round")),
                "confidence_mean": float(match.group("confidence")),
                "need_mean": float(match.group("need")),
                "slots": slot_values,
                "source_file": str(path.name),
            }
        )
    return rows


def ensure_consistent_slot_count(rows):
    slot_counts = sorted({len(row["slots"]) for row in rows})
    if not slot_counts:
        raise ValueError("No CCR slot records found in the provided logs.")
    if len(slot_counts) != 1:
        raise ValueError(f"Inconsistent slot counts found: {slot_counts}")
    return slot_counts[0]


def build_round_client_tensor(rows, slot_count):
    rounds = sorted({row["round"] for row in rows})
    clients = sorted({row["client"] for row in rows})
    round_to_client_slots = {}
    for round_id in rounds:
        client_map = {}
        for row in rows:
            if row["round"] != round_id:
                continue
            client_map[row["client"]] = np.asarray(row["slots"], dtype=np.float64)
        if len(client_map) == len(clients):
            round_to_client_slots[round_id] = client_map
    return rounds, clients, round_to_client_slots


def compute_summary(rows, slot_count, flat_threshold):
    rounds, clients, round_to_client_slots = build_round_client_tensor(rows, slot_count)
    if not round_to_client_slots:
        raise ValueError("No rounds contain complete client coverage for CCR slot analysis.")

    cross_client_std_per_round_slot = []
    cross_slot_std_per_round_client = []
    slot_means_by_client = defaultdict(list)
    client_means_by_slot = defaultdict(list)

    for round_id, client_map in round_to_client_slots.items():
        slot_matrix = np.stack([client_map[client_id] for client_id in clients], axis=0)
        cross_client_std = np.std(slot_matrix, axis=0)
        cross_client_std_per_round_slot.append((round_id, cross_client_std))

        for slot_idx in range(slot_count):
            for client_id in clients:
                client_means_by_slot[(slot_idx, client_id)].append(client_map[client_id][slot_idx])

        for client_id in clients:
            slot_std = float(np.std(client_map[client_id]))
            cross_slot_std_per_round_client.append((round_id, client_id, slot_std))
            slot_means_by_client[client_id].append(client_map[client_id])

    mean_cross_client_std_by_slot = np.mean(
        np.stack([item[1] for item in cross_client_std_per_round_slot], axis=0),
        axis=0,
    )
    mean_cross_slot_std_by_client = {
        client_id: float(np.mean([row[2] for row in cross_slot_std_per_round_client if row[1] == client_id]))
        for client_id in clients
    }
    mean_slot_vector_by_client = {
        client_id: np.mean(np.stack(slot_means_by_client[client_id], axis=0), axis=0).tolist()
        for client_id in clients
    }
    mean_client_value_by_slot = {
        f"slot_{slot_idx}": {
            f"client_{client_id}": float(np.mean(client_means_by_slot[(slot_idx, client_id)]))
            for client_id in clients
        }
        for slot_idx in range(slot_count)
    }

    overall_client_std = float(np.mean(mean_cross_client_std_by_slot))
    overall_slot_std = float(np.mean(list(mean_cross_slot_std_by_client.values())))

    if overall_client_std < flat_threshold and overall_slot_std < flat_threshold:
        interpretation = (
            "Both cross-client and cross-slot confidence variation are weak. "
            "Current chunk-based slot construction is likely too flat."
        )
    elif overall_client_std >= flat_threshold and overall_slot_std < flat_threshold:
        interpretation = (
            "Confidence varies across clients but not across slots. "
            "Current routing signal looks client-level rather than slot-level."
        )
    elif overall_client_std < flat_threshold and overall_slot_std >= flat_threshold:
        interpretation = (
            "Confidence varies across slots inside each client, but clients look similar to each other. "
            "Slots may capture generic structure, not client-specific complementary expertise."
        )
    else:
        interpretation = (
            "Confidence varies across both clients and slots. "
            "Current v3.1 routing has non-trivial structure and is worth further diagnosis."
        )

    return {
        "num_records": len(rows),
        "num_complete_rounds": len(round_to_client_slots),
        "rounds": sorted(round_to_client_slots.keys()),
        "clients": clients,
        "slot_count": slot_count,
        "flat_threshold": flat_threshold,
        "overall_mean_cross_client_std": overall_client_std,
        "overall_mean_cross_slot_std": overall_slot_std,
        "mean_cross_client_std_by_slot": mean_cross_client_std_by_slot.tolist(),
        "mean_cross_slot_std_by_client": mean_cross_slot_std_by_client,
        "mean_slot_vector_by_client": mean_slot_vector_by_client,
        "mean_client_value_by_slot": mean_client_value_by_slot,
        "interpretation": interpretation,
    }, cross_client_std_per_round_slot, cross_slot_std_per_round_client


def write_timeseries_csv(rows, slot_count, output_dir):
    csv_path = output_dir / "confidence_timeseries.csv"
    fieldnames = ["round", "client", "confidence_mean", "need_mean", "source_file"] + [
        f"slot_{idx}" for idx in range(slot_count)
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["round"], item["client"])):
            out_row = {
                "round": row["round"],
                "client": row["client"],
                "confidence_mean": row["confidence_mean"],
                "need_mean": row["need_mean"],
                "source_file": row["source_file"],
            }
            for idx, value in enumerate(row["slots"]):
                out_row[f"slot_{idx}"] = value
            writer.writerow(out_row)
    return csv_path


def write_cross_client_csv(cross_client_std_per_round_slot, slot_count, output_dir):
    csv_path = output_dir / "cross_client_std_by_round.csv"
    fieldnames = ["round"] + [f"slot_{idx}_std" for idx in range(slot_count)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for round_id, slot_std in cross_client_std_per_round_slot:
            row = {"round": round_id}
            for idx, value in enumerate(slot_std.tolist()):
                row[f"slot_{idx}_std"] = value
            writer.writerow(row)
    return csv_path


def write_cross_slot_csv(cross_slot_std_per_round_client, output_dir):
    csv_path = output_dir / "cross_slot_std_by_round_client.csv"
    fieldnames = ["round", "client", "slot_std"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for round_id, client_id, slot_std in cross_slot_std_per_round_client:
            writer.writerow(
                {
                    "round": round_id,
                    "client": client_id,
                    "slot_std": slot_std,
                }
            )
    return csv_path


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir does not exist: {log_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else log_dir / "ccr_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    client_logs = sorted(log_dir.glob("client*.log"))
    if not client_logs:
        raise FileNotFoundError(f"No client*.log files found under {log_dir}")

    rows = []
    for path in client_logs:
        rows.extend(parse_client_log(path))

    slot_count = ensure_consistent_slot_count(rows)
    summary, cross_client_std_per_round_slot, cross_slot_std_per_round_client = compute_summary(
        rows,
        slot_count,
        args.flat_threshold,
    )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ts_path = write_timeseries_csv(rows, slot_count, output_dir)
    cc_path = write_cross_client_csv(cross_client_std_per_round_slot, slot_count, output_dir)
    cs_path = write_cross_slot_csv(cross_slot_std_per_round_client, output_dir)

    print("CCR log analysis finished.")
    print(f"log_dir: {log_dir}")
    print(f"records: {summary['num_records']}")
    print(f"complete_rounds: {summary['num_complete_rounds']}")
    print(f"slot_count: {summary['slot_count']}")
    print(f"overall_mean_cross_client_std: {summary['overall_mean_cross_client_std']:.6f}")
    print(f"overall_mean_cross_slot_std: {summary['overall_mean_cross_slot_std']:.6f}")
    print(f"interpretation: {summary['interpretation']}")
    print(f"summary_json: {summary_path}")
    print(f"timeseries_csv: {ts_path}")
    print(f"cross_client_csv: {cc_path}")
    print(f"cross_slot_csv: {cs_path}")


if __name__ == "__main__":
    main()
