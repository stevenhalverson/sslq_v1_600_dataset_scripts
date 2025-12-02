# SSLQ Version 1.600 — Dataset Scripts

This repository contains the **annotation schema** and **data-cleaning scripts** used to prepare the public release of [SSLQ Version 1.600](https://huggingface.co/datasets/shalvers/SSLQ-Version-1.600) as well as the creation of LoRA models based on the SSLQ dataset [sslq_v1_600_dataset_scripts](https://github.com/stevenhalverson/sslq_v1_600_dataset_scripts).

---

### 📁 Contents

- `/schema/sslq_schema_v1_0.xml` — Label Studio interface definition  
- `/scripts/` — Python utilities for cleaning and renumbering metadata to ensure proper Hugging Face viewer compatibility
- `/image_model_training/` — Python utilities for pruning and sorting images and creating .txt files to train an image model, LoRA or possibly fine-tune
- `/requirements/` — Minimal runtime dependencies for these utilities

---

### 🧩 Environments

Two environments were used during development of the dataset:

- **Label Studio environment** – for annotation and schema testing.  
  See [`requirements/requirements_ls.txt`](requirements/requirements_ls.txt)  
  *(Full export from the local Label Studio instance; mainly for documentation.)*

- **Hugging Face / Upload environment** – for dataset cleaning, metadata generation, and Hub upload.  
  See [`requirements/requirements_hf.txt`](requirements/requirements_hf.txt)  
  *(Lightweight environment focused on pandas, pillow, and huggingface_hub utilities, for documentation.)*

---

### 🤖 Reverse-Prompt Generation (Vertex AI)

The script [`gemini_reverse_prompt_batch.py`](scripts/gemini_reverse_prompt_batch.py) uses **Google Vertex AI – Gemini 2.5 Flash Lite** to create short, structured “reverse prompts” for each image in the dataset.  
It was used to produce the `llm_description` column included in the dataset’s metadata.

**Key features:**
- Safe, resumable generation (skips already-processed files)  
- Exponential backoff on transient API errors  
- Clear, structured image-to-text prompting style  
- No personal or private data used  

To run this script, you’ll need access to a Vertex AI project and the following minimal environment:

```bash
pip install google-cloud-aiplatform vertexai

```

⚖️ License & Attribution

Author: Steven Halverson
License: MIT (for scripts), CC-BY-4.0 (for dataset)