import os
from pathlib import Path

from dotenv import load_dotenv
from ragbuilder import RAGBuilder

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "privacy_statement_en-us.txt"
PROJECT_DIR = ROOT_DIR / "rag_project"


def require_openai_key() -> None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it in .env before running this script."
        )


def build_and_save(n_trials: int = 5) -> None:
    # Override any pre-set empty shell vars with local .env values.
    load_dotenv(ROOT_DIR / ".env", override=True)
    require_openai_key()

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_FILE}")

    print(f"Building RAG pipeline from: {DATA_FILE}")
    builder = RAGBuilder.from_source_with_defaults(
        input_source=str(DATA_FILE),
        n_trials=n_trials,
    )

    results = builder.optimize()

    print("\nOptimization summary:")
    print(results.summary())

    sample_question = "What personal information is collected and for what purposes?"
    answer = results.invoke(sample_question)

    print("\nSample question:")
    print(sample_question)
    print("\nSample answer:")
    print(answer)

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    builder.save(str(PROJECT_DIR))
    print(f"\nSaved optimized project to: {PROJECT_DIR}")


if __name__ == "__main__":
    build_and_save(n_trials=1)
