# 📚 EPUB Novel Summarizer

Compress a massive EPUB novel — 2,000+ chapters — down to ~1/10th of its original length using a **local AI model via [Ollama](https://ollama.com)**. No API keys, no cloud, no cost. Runs entirely on your machine.

Built for readers who want to quickly catch up on long web novels (xianxia, wuxia, isekai, etc.) without reading millions of words.

---

## ✨ Features

- **Chapter-aware summarization** — reads each chapter individually and targets a word count proportional to the original (default: 10%)
- **Preserves chapter titles** — the original chapter name is kept so you can always cross-reference the source
- **Rolling context window** — passes summaries of the 3 most recent chapters as context so each new summary stays coherent and connected
- **Smart chunking** — chapters exceeding the model's context limit are automatically split, summarized in parts, then merged into one cohesive summary
- **Auto-resume** — progress is saved after every chapter to a `.progress.json` file; if interrupted, the script picks up exactly where it left off
- **Chapter range filtering** — process only a slice of the book with `--start-chapter` / `--end-chapter`
- **Dry-run mode** — preview all chapters and their word counts without calling the AI
- **Output as EPUB** — produces a clean, valid EPUB file with a table of contents, readable in any e-reader

---

## 🖥️ Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### Python dependencies

```bash
pip install ebooklib beautifulsoup4 requests tqdm
```

---

## ⚙️ Setup

**1. Install and start Ollama**

```bash
# Install: https://ollama.com/download
ollama serve
```

**2. Pull a model**

```bash
ollama pull llama3          # recommended default
ollama pull qwen2.5:7b      # best for Asian-language novels
ollama pull mistral         # lighter, good for low-RAM machines
```

**3. Clone this repo and install dependencies**

```bash
git clone https://github.com/your-username/epub-summarizer.git
cd epub-summarizer
pip install ebooklib beautifulsoup4 requests tqdm
```

---

## 🚀 Usage

### Basic — summarize entire book to 10% of original length

```bash
python epub_summarizer.py "My Novel.epub" "My Novel - Summary.epub"
```

### Custom ratio and model

```bash
python epub_summarizer.py input.epub output.epub --ratio 0.15 --model qwen2.5:7b
```

### Process only a range of chapters

```bash
python epub_summarizer.py input.epub output.epub --start-chapter 0 --end-chapter 100
```

### Force Vietnamese output language

```bash
python epub_summarizer.py input.epub output.epub --lang vi
```

### Preview chapters without calling AI

```bash
python epub_summarizer.py input.epub output.epub --dry-run
```

### Use a remote Ollama instance

```bash
python epub_summarizer.py input.epub output.epub --ollama-url http://192.168.1.10:11434/api/generate
```

---

## 🔧 All Options

| Argument | Default | Description |
|---|---|---|
| `input` | *(required)* | Path to the source EPUB file |
| `output` | *(required)* | Path for the summarized EPUB output |
| `--model` | `llama3` | Ollama model name to use |
| `--ratio` | `0.1` | Target length as a fraction of original (0.1 = 10%) |
| `--lang` | `auto` | Output language: `auto`, `vi`, or `en` |
| `--start-chapter` | `0` | First chapter to process (0-indexed) |
| `--end-chapter` | *(end of book)* | Last chapter to process (exclusive) |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `--dry-run` | `False` | List chapters and word counts without calling AI |

---

## 🧠 How It Works

```
Input EPUB
    │
    ├── Extract chapters (skip pages < 50 words)
    │
    └── For each chapter:
            │
            ├── Count words  (e.g. 4,000 words)
            ├── Calculate target  (e.g. 400 words at 10%)
            ├── Build prompt with:
            │       - Rolling context from last 3 summaries (~600 words)
            │       - Full chapter text
            │         (auto-chunked if chapter > 7,000 words)
            ├── Call Ollama → get summary
            ├── Save to progress file
            └── Add summary to context window

Output EPUB
    ├── Original chapter titles preserved
    ├── Word count stats per chapter (original → summary)
    └── Full table of contents
```

---

## 📁 Output Format

Each chapter in the output EPUB looks like this:

```
Chapter 142: The Final Battle at Dragon Peak

📖 Original: 3,840 words → Summary: 391 words

Wei Chen arrived at the summit to find Elder Gu already waiting.
Without preamble, Gu unleashed his full cultivation base — a pressure
that cracked the stone beneath their feet. Wei Chen countered with the
Heaven-Splitting technique he had mastered in the Hidden Valley...
```

---

## 💾 Resume After Interruption

Every completed chapter is immediately saved to `output.epub.progress.json`. If the process is stopped for any reason — power loss, Ollama crash, manual cancel — simply re-run the exact same command. Already-completed chapters are skipped and the run continues from where it stopped.

The progress file is automatically deleted once all chapters are finished.

---

## 🌏 Model Recommendations

| Model | Best For | RAM Required |
|---|---|---|
| `qwen2.5:7b` | Chinese/Vietnamese novels, strong contextual understanding | ~6 GB |
| `llama3` | English novels, fast and stable | ~5 GB |
| `mistral` | Low-resource machines, decent quality | ~4 GB |
| `llama3:70b` | Highest quality summaries (slow) | ~40 GB |

For non-English novels, `qwen2.5` generally produces the most natural-sounding summaries.

---

## ⚠️ Limitations

- Quality depends on the Ollama model used — larger models produce better summaries
- Very plot-dense chapters may lose minor details at aggressive ratios (< 5%)
- The script processes chapters sequentially; parallel processing is not supported

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
