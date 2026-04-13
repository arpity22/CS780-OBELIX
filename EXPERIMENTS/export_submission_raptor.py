import argparse
import os
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def pick_weights(weights_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = weights_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Requested weights file not found: {p}")
        return p

    candidates = [
        weights_dir / "weights_submission_best.pth",
        weights_dir / "weights_eval_best.pth",
        weights_dir / "weights_best.pth",
        weights_dir / "weights.pth",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"No weights found in {weights_dir}. Expected one of: "
        "weights_submission_best.pth, weights_eval_best.pth, weights_best.pth, weights.pth"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OBELIX submission.zip from agent and weights.")
    parser.add_argument("--agent_src", type=str, default="agent_raptor_ppo.py")
    parser.add_argument("--weights_dir", type=str, default="submission_raptor_ppo")
    parser.add_argument("--weights_file", type=str, default=None)
    parser.add_argument("--submission_dir", type=str, default="submission_build")
    parser.add_argument("--zip_name", type=str, default="submission.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    agent_src = Path(args.agent_src)
    if not agent_src.exists():
        raise FileNotFoundError(f"Agent source file not found: {agent_src}")

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    selected_weights = pick_weights(weights_dir=weights_dir, explicit=args.weights_file)

    submission_dir = Path(args.submission_dir)
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    dst_agent = submission_dir / "agent.py"
    dst_weights = submission_dir / "weights.pth"

    shutil.copy2(agent_src, dst_agent)
    shutil.copy2(selected_weights, dst_weights)

    zip_path = Path(args.zip_name)
    if not zip_path.is_absolute():
        zip_path = Path.cwd() / zip_path

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.write(dst_agent, arcname="agent.py")
        zf.write(dst_weights, arcname="weights.pth")

    print(f"Selected weights: {selected_weights}")
    print(f"Submission folder: {submission_dir}")
    print(f"Created zip: {zip_path}")


if __name__ == "__main__":
    main()
