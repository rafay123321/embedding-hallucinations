# Embedding Hallucinations

This repository supports research and experimentation around understanding and mitigating hallucinations in embeddings — specifically how embeddings can fail to capture human-like understanding.

## 🧠 Objectives

1. **Compare Embeddings**: Measure similarity between sentences using cosine similarity (or other metrics like dot product, Euclidean distance).
2. **Fine-Tune Embeddings**: Fine-tune SentenceTransformer models to reduce hallucinations and improve semantic understanding.

---

## 📁 Folder Structure

```
.
├── data/                        # Training, validation, and test data
├── fine-tuning/                
│   ├── embedding-fine-tune.py  # Fine-tunes embedding models
│   └── eval.py                 # Evaluates and compares model embeddings
├── outputs/                    # Outputs of similarity scoring between sentence pairs
├── results/                    # Results of evaluation and comparisons
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🔧 Environment Setup

You can create an environment using any of the following:

### Option 1: Using virtualenv

```bash
python -m venv halluc-env
source halluc-env/bin/activate  # On Windows: halluc-env\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
conda create -n halluc-env python=3.10
conda activate halluc-env
pip install -r requirements.txt
```

### Option 3: Using uv (Ultra fast package installer)

```bash
uv venv halluc-env
source halluc-env/bin/activate
uv pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Fine-Tune Embedding Models

Fine-tune a SentenceTransformer model using the provided training data to reduce hallucinations:

```bash
python ./fine-tuning/embedding-fine-tune.py
```

### 2. Evaluate Embeddings

Compare a fine-tuned model against a foundational model using evaluation datasets:

```bash
python ./fine-tuning/eval.py
```

This will output evaluation results in the `results/` directory.

### 3. Compare Sentence Similarity

Use the sentence similarity comparison utility to find the semantic similarity between any two sentences. Cosine similarity is used by default.

Results are stored in the `outputs/` directory.

---

## 🔍 Similarity Metrics

Currently implemented:  
- ✅ Cosine Similarity (default)

You can easily switch to other metrics such as:
- Dot Product
- Euclidean Distance

These metrics are applied on the sentence embeddings generated using SentenceTransformer models.

---

## 📊 Data & Results

- `data/`: Contains training, validation, and test sets used for fine-tuning.
- `results/`: Contains evaluation output comparing foundational and fine-tuned models.
- `outputs/`: Contains similarity scores between sentence pairs.

---

## 🧪 Research Focus

This project is part of research for the paper:

**"Hallucination by Design: How Embeddings Fail Understanding Human Language"**

It explores:
- Where and why embeddings hallucinate
- How fine-tuning helps mitigate such hallucinations
- Benchmarks for measuring improvements

---

## 📄 License

This project is released under the MIT License.

---

## 🤝 Contributing

Feel free to fork, experiment, and contribute via pull requests or discussions.

---

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/)
- The open-source community supporting transparency in embedding evaluation and interpretability