# Tokenization Deep Dive

[![Live demo](https://img.shields.io/badge/demo-GitHub%20Pages-22d3ee?style=flat-square&logo=github&logoColor=white)](https://gaurav36.github.io/tokenization-explanation/)
[![Pages](https://img.shields.io/github/actions/workflow/status/gaurav36/tokenization-explanation/pages.yml?label=pages&style=flat-square)](https://github.com/gaurav36/tokenization-explanation/actions/workflows/pages.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/packaging-uv-DE5FE9?style=flat-square)](https://docs.astral.sh/uv/)
[![tokenizers](https://img.shields.io/badge/tokenizers-0.23.1-FFD21E?style=flat-square)](https://github.com/huggingface/tokenizers)
[![sentencepiece](https://img.shields.io/badge/sentencepiece-0.2.2-4285F4?style=flat-square)](https://github.com/google/sentencepiece)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)

Interactive course for AI engineers: a static **token lab** in the browser, then three Jupyter notebooks that train BPE, WordPiece, and SentencePiece from scratch before using current PyPI libraries via **uv**.

**[Open the live site](https://gaurav36.github.io/tokenization-explanation/)** — no install. All playgrounds run in your browser.

Deep links for class chat:

- [BPE lab](https://gaurav36.github.io/tokenization-explanation/?tab=bpe#algo-labs)
- [WordPiece lab](https://gaurav36.github.io/tokenization-explanation/?tab=wordpiece#algo-labs)
- [SentencePiece lab](https://gaurav36.github.io/tokenization-explanation/?tab=sentencepiece&sp=unigram#algo-labs)

## What you get

| Surface | Runs where | What students do |
|---|---|---|
| [Live site](https://gaurav36.github.io/tokenization-explanation/) | GitHub Pages (static HTML/JS) | Pipeline, Unicode/UTF-8, four-way splitter, OOV trap, UTF-8 inspector, BPE walkthrough, in-browser BPE / WordPiece / SentencePiece, compare row, predict-then-reveal, vocab-vs-context |
| Jupyter notebooks | Local Python 3.12 + uv | Same algorithms from scratch, then HuggingFace `tokenizers` 0.23.1, `sentencepiece` 0.2.2, `tiktoken` 0.14.0 |

The website does **not** need Python. Notebooks do **not** run on GitHub Pages.

## Syllabus

1. **Tokenization** — what tokens are and why language models need them
2. **How tokenization works** — the path from raw text to token IDs and back
3. **Unicode and UTF-8** — graphemes, code points, and bytes
4. **Four-way lab** — compare word, subword, character, and byte token counts
5. **Word level** — short sequences, fixed vocabularies, and the `[UNK]` problem
6. **Subword level** — reusable pieces learned with BPE, WordPiece, or SentencePiece
7. **How BPE works** — pair counts, ordered merges, stopping rules, and fallback
8. **Algorithm labs** — train and compare BPE, WordPiece, and SentencePiece in the browser
9. **Character level** — Unicode code points and visible grapheme clusters
10. **Byte level** — total UTF-8 coverage, byte fallback, and BPE on top of bytes
11. **Vocabulary and context** — parameters, vocabulary size, sequence length, and context length
12. **Visuals** — diagrams of the pipeline, granularity, training, and decoding
13. **Glossary** — definitions of the lesson's key tokenizer terms
14. **Notebooks** — build BPE, WordPiece, and SentencePiece in Jupyter

No prior background assumed. The site opens with a plain-English on-ramp and a suggested
reading path for complete beginners.

## Live playgrounds

The live site includes a four-way split of the same string (word, naive subword,
character, and UTF-8 bytes) plus one shared corpus for comparing **BPE**,
**WordPiece**, and **SentencePiece** without leaving the page.

| Lab | What it teaches |
|---|---|
| Four-way splitter | How token *count* explodes as units get smaller |
| OOV trap | Word-level `[UNK]` when a type was never in the vocab (predict, then reveal) |
| UTF-8 inspector | Graphemes vs code points vs bytes (emoji, Indic, Japanese) |
| Length bars | Vocab size vs sequence length (**naive 3-char is labeled — not BPE**) |
| BPE tab | Max pair frequency, `</w>` end-of-word mark, round-trip decode |
| WordPiece tab | Score `freq(ab)/(freq(a)·freq(b))`, greedy `##` encode |
| SentencePiece tab | Unigram Viterbi with `▁`, or BPE with no whitespace split |
| Compare row | Same probe under all three algorithms at once |

Default teaching corpus is the Sennrich toy set. WordPiece on that set encodes `lowest` as `low` + `##est`.

Projector tip: use the **Light** theme toggle in the nav.

## Teaching diagrams

Full-page SVG schematics (Token Lab cyan on light paper) — better for projectors than mermaid fences:

| Diagram | URL |
|---|---|
| Gallery | https://gaurav36.github.io/tokenization-explanation/diagrams/ |
| Tokenizer pipeline | [01-pipeline.html](https://gaurav36.github.io/tokenization-explanation/diagrams/01-pipeline.html) |
| Granularity layers | [02-granularity.html](https://gaurav36.github.io/tokenization-explanation/diagrams/02-granularity.html) |
| BPE train loop | [03-bpe-train.html](https://gaurav36.github.io/tokenization-explanation/diagrams/03-bpe-train.html) |
| BPE vs WordPiece | [04-bpe-vs-wordpiece.html](https://gaurav36.github.io/tokenization-explanation/diagrams/04-bpe-vs-wordpiece.html) |
| Detokenize marks | [05-detokenize-marks.html](https://gaurav36.github.io/tokenization-explanation/diagrams/05-detokenize-marks.html) |

Local path: `site/diagrams/`.

## Quickstart (notebooks)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. Do not use pip.

```bash
git clone https://github.com/gaurav36/tokenization-explanation.git
cd tokenization-explanation
uv sync
uv run jupyter lab notebooks
```

Work through the notebooks in order:

| Notebook | Topic |
|---|---|
| [`notebooks/01_bpe.ipynb`](notebooks/01_bpe.ipynb) | Frequency BPE from scratch (Sennrich), HuggingFace `BpeTrainer` (course corpus), then `tiktoken` |
| [`notebooks/02_wordpiece.ipynb`](notebooks/02_wordpiece.ipynb) | Likelihood WordPiece from scratch, `##` continuation, `WordPieceTrainer` |
| [`notebooks/03_sentencepiece.ipynb`](notebooks/03_sentencepiece.ipynb) | Unigram Viterbi intuition, then `sentencepiece` 0.2.2 (`return_type=`, not deprecated `out_type`) |

Each production trainer uses [`data/tiny_corpus.txt`](data/tiny_corpus.txt). Written models go to `artifacts/` (gitignored).

**Network notes**

- First `tiktoken.get_encoding(...)` **downloads** encoding files — do that once on a network before an offline lab.

Local site (optional; the Pages URL is enough for class):

```bash
uv run python -m http.server 8000 --directory site
```

Then open `http://localhost:8000`. You can also open `site/index.html` directly (`file://` works).

## Regression checks

```bash
node tests/golden_algos.mjs
uv run python scripts/build_notebooks.py
```

## Project layout

```
tokenization-explanation/
├── site/                         # Static website deployed to GitHub Pages
│   ├── index.html                # Complete 14-section lesson
│   ├── css/main.css              # Layout and visual styles
│   ├── js/                       # Labs, tokenizers, and interactions
│   ├── diagrams/                 # Five teaching diagrams
│   ├── assets/                   # Favicon and social preview image
│   └── .nojekyll                 # Publish static files without Jekyll
├── notebooks/                    # BPE, WordPiece, and SentencePiece lessons
│   ├── 01_bpe.ipynb
│   ├── 02_wordpiece.ipynb
│   └── 03_sentencepiece.ipynb
├── data/                         # Small corpora used by labs and notebooks
│   ├── sennrich_toy.txt
│   └── tiny_corpus.txt
├── scripts/build_notebooks.py    # Rebuild generated notebook content
├── tests/golden_algos.mjs        # Tokenizer regression checks
├── .github/workflows/pages.yml   # GitHub Pages deployment workflow
├── pyproject.toml                # Python dependencies and project metadata
├── uv.lock                       # Reproducible Python dependency versions
├── LICENSE
└── README.md
```

## Libraries (pinned via uv)

Resolved from PyPI; see `uv.lock`.

| Package | Role |
|---|---|
| `tokenizers` 0.23.1 | HuggingFace BPE and WordPiece trainers |
| `sentencepiece` 0.2.2 | Google SentencePiece (pybind11 API) |
| `tiktoken` 0.14.0 | Production byte-level BPE (OpenAI models) |
| `jupyterlab` 4.6.3, `ipykernel`, `ipywidgets` | Notebooks and live widgets |

SentencePiece 0.2.2: use `encode(..., return_type=str)` and `SentencePieceProcessor.from_file(...)`. Do not use deprecated `out_type` or `EncodeAsImmutableProto`.

## GitHub Pages

The site is static. GitHub Actions (`.github/workflows/pages.yml`) uploads the `site/` folder on every push to `main`. No Jekyll (`.nojekyll`).

To publish the live site:

1. Open the repository on GitHub and go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.
3. Open the **Actions** tab and run the **Deploy GitHub Pages** workflow, or push a commit to `main`.
4. After the workflow succeeds, open <https://gaurav36.github.io/tokenization-explanation/>.

For a GitHub Free account, the simplest option is to make the repository **Public** under
**Settings → General → Danger Zone → Change repository visibility**. Publishing Pages from a
private repository requires a GitHub plan that supports private-repository Pages. After renaming
or changing the visibility of a repository, rerun the Pages workflow and allow a few minutes for
the new URL to become available.

If the site still returns `404`, check that the repository name is exactly
`tokenization-explanation`, the default branch is `main`, Pages uses **GitHub Actions**, and the
latest Pages workflow completed successfully.

Notebooks, `uv`, and trained `.model` files under `artifacts/` are **not** part of Pages.

## License

[MIT](LICENSE)

