# KG-Inspect

**Knowledge Graph, RAG & Visual Inspection Toolkit**

KG-Inspect is a **Python-based CLI toolkit** for building and querying a **Neo4j Knowledge Graph** integrated with **Retrieval-Augmented Generation (LightRAG)** and **industrial visual inspection models**.

---

## ✨ Key Features

### 🧠 Knowledge Graph (Neo4j)

- Manage datasets, categories, and complex relationships.
- Programmatically insert **custom Knowledge Graphs** via JSON.
- Designed specifically for structured industrial and technical knowledge.

### 📚 LightRAG Integration

- Document ingestion (TXT, PDF, etc.).
- Multiple query modes: `naive`, `local`, `global`, and `hybrid` (recommended).

### 🏭 Inspection Pipeline

- **Text-only inspection** using Knowledge Graph + RAG.
- **Image-assisted inspection** using CNN/Vision models and VLMs.

### 🖥 CLI-first Design

- Reproducible, scriptable workflows via the `kg-inspect` command.

---

## 📁 Project Structure

```text
kg_inspect/
├── cli/                 # CLI commands (lightrag, models)
├── rag/                 # LightRAG configuration and managers
├── utils/               # Utilities (DB checks, helpers)
├── gradio/              # Gradio-based UI
│   └── app.py           # Gradio application entry point
├── __main__.py          # CLI entry point
Makefile                 # Development helper commands
pyproject.toml           # Project metadata & dependencies
.env-temp                # Environment variable template
README.md
```

---

## 🧠 Pretrained Models (Required for Vision Inspection)

This repository **does NOT include any pretrained model files** and **does NOT ship with a `pretrained_models/` folder in the GitHub source code**.

Because the pretrained checkpoints are **large, experimental, and non-standard**, they are distributed via **GitHub Releases** instead of being committed to the repository.

---

### 📦 Download Pretrained Models

Pretrained models are provided via GitHub **[Releases](https://github.com/your-username/your-repo/releases)**.

> Look for a release asset such as `pretrained_models.zip` (or a similarly named archive).

---

### ✅ How to set up pretrained models

1. Go to the GitHub **[Releases](https://github.com/your-username/your-repo/releases)** page
2. Download the pretrained model archive  
   (for example: `pretrained_models.zip`)
3. Extract it **into the project root directory**

After extraction, your directory structure should look like this:

```text
your-repository-name/
├── kg_inspect/
├── pretrained_models/                 # ← extracted from GitHub Release (NOT in repo)
│   ├── Convnext_tiny/
│   │   └── convnext_tiny.pth
│   └── Cutpaste/
│       ├── bottle/
│       ├── capsule/
│       └── ...
├── pyproject.toml
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Environment Setup

You can use the provided Makefile to automate the setup:

```bash
# Create a virtual environment (Python 3.11)
make create-venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\Activate.ps1 # Windows

# Install the project in editable mode
make setup
```

### 2. Configure Neo4j

```bash
cp .env-temp .env
```

Edit `.env` with your `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.

---

## 🛠 Makefile Usage

The `Makefile` includes several utility commands for development:

| Command                     | Description                                               |
| :-------------------------- | :-------------------------------------------------------- |
| `make setup`                | Install the package in editable mode (`pip install -e .`) |
| `make lint`                 | Format code using `black` and `isort`                     |
| `make create-venv`          | Create a Python 3.11 virtual environment                  |
| `make install-requirements` | Install dependencies from `requirements.txt`              |
| `make create-requirements`  | Force-regenerate `requirements.txt` using `pipreqs`       |
| `make build`                | Build the distribution packages (wheel and sdist)         |
| `make clean`                | Remove build artifacts, cache files, and temp docs        |

---

## 🧪 CLI Usage

### LightRAG & Knowledge Graph

```bash
# Test connection
kg-inspect lightrag test-connection-kg

# Ingest data
kg-inspect lightrag insert-doc data/manual.pdf
kg-inspect lightrag insert-custom-kg data/schema.json

# Query the system
kg-inspect lightrag query "Search query" --mode hybrid

# Insert all custom kg short script
python kg_inspect/data/custom_kg/install_all_custom_kg.py
```

### Inspection Models

```bash
# Run visual inspection on images
kg-inspect models run \
  -q "Analyze these surfaces for cracks" \
  -i data/img.jpg
```

### 🖼️ UI Access

To launch the Gradio interface:

```bash
gradio ./kg_inspect/gradio/app.py
```

---

## 🧯 Troubleshooting

- **Neo4j Connection:** Use `kg-inspect lightrag test-connection-kg` to verify your `.env` settings.
- **Missing Models:** If inspection fails, ensure you have downloaded the weights from the **Releases** page.
- **CUDA Errors:** If you don't have a GPU, ensure `DEVICE=cpu` is set in your `.env` file.

---

## 🗺 Roadmap

- Example `custom_kg.json` schemas.
- End-to-end 5-minute quickstart demo.
- Docker support for easy deployment.
- Expanded visual inspection benchmarks.
