import os
import subprocess
import sys
from pathlib import Path


def _find_project_python() -> str:
    """
    Chọn Python interpreter phù hợp để chạy DVC stages.

    Khi chạy local trên Windows, DVC có thể resolve nhầm `python` global.
    Nếu repo có `.venv`, ta ưu tiên dùng Python trong `.venv`.
    Khi chạy trên GitHub Actions hoặc môi trường không có `.venv`, dùng Python hiện tại.
    """
    root = Path(__file__).resolve().parent

    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return sys.executable


def main() -> int:
    """
    Chạy một Python script bằng interpreter phù hợp.

    Ví dụ:
        python run_stage.py generate_data.py
        python run_stage.py src/train.py
    """
    if len(sys.argv) < 2:
        print("Usage: python run_stage.py <script.py> [args...]")
        return 1

    python_executable = _find_project_python()
    script_and_args = sys.argv[1:]

    print(f"Running with Python: {python_executable}")
    print(f"Running command: {script_and_args}")

    completed = subprocess.run(
        [python_executable, *script_and_args],
        check=False,
        cwd=Path(__file__).resolve().parent,
    )

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())