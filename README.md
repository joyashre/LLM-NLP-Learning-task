# Applied NLP & LLM Techniques 🚀

A comprehensive collection of advanced Natural Language Processing and Large Language Model (LLM) implementations. This repository contains practical solutions ranging from classical NLP algorithms to modern LLM fine-tuning and Retrieval-Augmented Generation (RAG).

## 📚 Tasks Overview

### Task 3: RAG-based QA System
* **Description:** Implemented a Retrieval-Augmented Generation (RAG) system for question-answering on live webpages. The system chunks web data, stores it in a vector database, and retrieves relevant context to generate factual answers.
* **Tech:** `LangChain`, `FAISS`, `HuggingFaceEmbeddings`, Mistral-7B.

### Task 4: Part-of-Speech (POS) Tagging
* **Description:** Developed Part-of-Speech taggers using two distinct approaches:
  1. **Rule-based method:** Utilizing Regular Expressions for pattern matching.
  2. **N-gram Language Modeling:** Training Unigram and Bigram taggers (with backoff strategies) using the Penn Treebank corpus.
* **Tech:** `NLTK`.

### Task 5: Context-Free Grammar (CFG) & Parsing
* **Description:** Designed a custom Context-Free Grammar (CFG) for the **Bengali** language. Constructed and visualized constituency parse trees for complex sentences following the SOV (Subject-Object-Verb) structure.
* **Tech:** `NLTK`, `svgling` (for graphical tree visualization).

### Task 6: Word Sense Disambiguation (WSD)
* **Description:** Resolved lexical ambiguity (determining the correct meaning of a word based on its context) using:
  1. A custom Dictionary-based method computing WordNet definition overlaps.
  2. NLTK's built-in Lesk Algorithm.
* **Tech:** `NLTK`, WordNet.

### Task 7: LLM Fine-Tuning (Causal LM)
* **Description:** Fine-tuned a lightweight Causal Language Model (`distilgpt2`) specifically for the Word Sense Disambiguation task. Constructed a custom dataset and formatted it for instruction-based Supervised Fine-Tuning.
* **Tech:** `HuggingFace Transformers`, `trl` (SFTTrainer), `datasets`.

### Task 8: Coreference Resolution & Prompt Engineering
* **Description:** Leveraged a Causal LM (`TinyLlama-1.1B`) for coreference resolution to identify pronoun antecedents. Evaluated model performance across three distinct prompting strategies:
  1. Zero-shot prompting.
  2. Few-shot prompting.
  3. Chain of Thought (CoT) reasoning.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Generative AI & LLMs:** HuggingFace `transformers`, `trl`, LangChain
* **Vector Search:** FAISS
* **Classical NLP:** NLTK
* **Visualization:** `svgling`, IPython

## 🚀 Getting Started
These notebooks are optimized for Google Colab. To run the scripts locally, install the required dependencies:

```bash
pip install -qU langchain langchain-community langchain-huggingface faiss-cpu beautifulsoup4 sentence-transformers huggingface_hub nltk transformers datasets trl accelerate svgling
