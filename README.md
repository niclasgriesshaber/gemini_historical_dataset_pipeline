# 📚 LLM Historical Dataset Pipeline

A powerful, end-to-end pipeline for extracting structured information from historical documents using Google's Gemini-2.0-Flash model. This pipeline enables historians and researchers to automatically convert PDFs to images and directly extract structured data into CSV—all with robust supervision and control.

## 🔍 Overview

### Core Steps
- **📄➡️🖼️ PDF to PNG Conversion**
  - Converts each PDF page to a high-quality PNG using pdf2image (Poppler).
- **🖼️➡️📊 Structured Data Extraction**
  - Uses Google's Gemini-2.0-Flash to analyze the page images and extract data.
  - Produces CSV and JSON outputs ready for analysis.

Whether you have text-based PDFs or image-only historical scans, this pipeline offers a flexible, powerful approach for extracting historical data.

## ⚙️ Requirements

- Miniconda (we'll help you install it!)
- Google Cloud API key with access to Gemini-2.0-Flash (we'll show you how to get it)

## 🚀 Complete Setup Guide

### Step 1: Install Miniconda ✅
- Visit [Miniconda Installation](https://docs.conda.io/en/latest/miniconda.html)
- Download Miniconda for your operating system (Python 3.8 or newer)
- Run the installer
  - Windows: Double-click the .exe file
  - Mac/Linux: Run:
    ```
    bash Miniconda3-latest-MacOSX-x86_64.sh
    ```
- Accept the license and default settings
- Close and reopen your terminal
- Verify installation by typing:
  ```
  conda --version
  ```

### Step 2: Install an IDE (Recommended) 💻
Choose one of these modern IDEs for the best development experience:
- [Visual Studio Code](https://code.visualstudio.com/) - Microsoft's powerful code editor
- [Windsurf](https://windsurf.io/) - A modern IDE focused on AI development
- [Cursor](https://cursor.sh/) - An AI-first code editor

### Step 3: Obtain API Keys 🔑
- **Google API Key**
  - Go to [MakerSuite Google API Key](https://makersuite.google.com/app/apikey)
  - Sign in with your Google account
  - Click Create API Key (click the gear/settings icon if you don't see this)
  - Copy the key (a long string of letters and numbers)

### Step 4: Download and Install the Pipeline 💻
- Clone this repository: `git clone [repository-url] && cd llm_historical_dataset_pipeline`
- Create and activate the conda environment: `conda env create -f environment.yml && conda activate llm_historical`
- Install PDF processing tool (Poppler):
  - Mac: `conda install -c conda-forge poppler`
  - Windows: `conda install -c conda-forge poppler`
  - Ubuntu/Debian: `conda install -c conda-forge poppler`

### Step 5: Set Up Your Environment ⚙️
- Create a folder called `config` in your pipeline directory (if it doesn't exist).
- Inside `config`, create a file named `.env`.
- Open `.env` in any text editor and add your API keys:
  ```
  # Google API key for Gemini
  GOOGLE_API_KEY=your_google_api_key_here
  ```
- Replace the placeholder with your own credential.

### Step 6: Prepare Your Workspace 📁
- Create a folder called `data/pdfs` in your pipeline directory.
- Place your PDF documents in `data/pdfs`.

## ✏️ Writing an Effective Extraction Prompt

Create `src/prompt.txt` with instructions:

```
You are a helpful research assistant analyzing historical documents. 
Please extract the following information from this page:

1. Extract these specific fields:
   - Field1 (<type>): <explanation>
   - Field2 (<type>): <explanation>

2. Format the output as a JSON object with these exact field names:
[
{
    "field1": "extracted value",
    "field2": "extracted value"
}
]

Important guidelines:
- If a field is not found, use null
- Maintain original spelling and capitalization
- Include any uncertain readings in "additional_information"
```

## 🛠️ Usage

1) Convert PDF to PNG and extract data with one single command
```
python src/gemini-2.0-flash.py --pdf document.pdf
```

## 📂 Output Structure

```
data/
├── pdfs/
│   └── your_document.pdf
└── csvs/
    └── your_document/
        ├── your_document.csv
        └── page_by_page/
            ├── PNG/
            └── JSON/
```

## 🎮 Supervision and Control Options

Change temperature parameter

```
python src/gemini-2.0-flash.py --pdf document.pdf --temperature 0.2
```

Continue from a specific page if the process stopped unexpectedly

```
python src/gemini-2.0-flash.py --pdf document.pdf --continue_from_page 5
```

## ✨ Features

- 🔄 Direct Information Extraction from document images
- 🛡️ Error Handling: Automatic retries
- 🔀 Flexible Processing: Single or multiple documents
- ⏸️ Resumable Runs: Continue from any page
- 📊 Structured Output: JSON + CSV

## 🔧 Troubleshooting

- **📄 PDF Conversion Fails**
  - Ensure the PDF is not password-protected or corrupted.
- **🔑 API Errors**
  - Verify your GOOGLE_API_KEY in config/.env.
- **💾 Memory Issues**
  - Process fewer PDFs at once.

## 📜 License

MIT License

## 📝 How to Cite

If you use this pipeline in your research, please cite:

```bibtex
@article{gg2025,
    title={Multimodal LLMs for OCR, OCR-Post-Correction, and Named Entity Recognition in Historical Documents},
    author={Greif, Gavin and Griesshaber, Niclas},
    year={2025},
    journal={arXiv preprint},
    url={https://arxiv.org/abs/forthcoming}
}
```

## 🛠️ Code Generation

This code was generated with assistance of:
- o1-Pro
- Cursor IDE
- Sonnet-3.7-Thinking

## 📧 Contact

Niclas Griesshaber, niclasgriesshaber@outlook.com, Gavin Greif, gavin.greif@history.ox.ac.uk