# Contributing

Thank you for contributing to this bilingual RAG project.

## Project Goals

- Keep answers grounded in source documents.
- Preserve bilingual quality (French and English).
- Keep retrieval behavior explainable and deterministic.

## Local Setup

1. Create/activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your key in `.env`:

```bash
OPENAI_API_KEY=...
```

4. Run the app:

```bash
c:\RAG\.conda\python.exe -m streamlit run st_app.py
```

## Coding Guidelines

- Keep changes small and focused.
- Prefer explicit, readable logic over clever abstractions.
- Preserve metadata fields used by citations (`source`, `section`, `page`, `chunk_index`).
- Do not remove fallback paths (for example reranker fallback) unless replacing them with an equivalent safety mechanism.

## Retrieval Principles

- Hybrid retrieval (BM25 + vector) is intentional:
  - BM25 helps legal keyword precision.
  - Vector retrieval improves semantic recall on paraphrased questions.
- Reranking is a quality layer, not a hard dependency. The app must still return results if the cross-encoder model is unavailable.

## Pull Request Checklist

- [ ] App starts locally.
- [ ] Both FR and EN queries still work.
- [ ] Citations still render with metadata.
- [ ] No secrets are committed.
- [ ] README/docs updated if behavior changed.

## Commit Style

Use clear, imperative commits, for example:

- `Improve reranker fallback messaging`
- `Add Render blueprint for deployment`
- `Refine FR prompt for exhaustive queries`
