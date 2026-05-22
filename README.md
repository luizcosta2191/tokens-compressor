# 🗜️ TokenShrink - Multilingual Prompt Compressor

TokenShrink is a lightweight, efficient web application designed to optimize and compress AI prompts before sending them to Large Language Models (LLMs) like GPT-4, Gemini, or Claude. By removing redundant greetings, excessive whitespaces, and unnecessary stop words, TokenShrink helps developers and daily AI users save money and reduce API latency.

The application features a fully dynamic, bilingual user interface supporting both **English** and **Portuguese**.

---

## 🚀 Features

* **Bilingual UI & Logic:** Toggle between English and Portuguese to dynamically update the interface and the text-cleaning algorithms.
* **Smart Trim:** Automatically strips conversational fluff, courtesies, and greetings (e.g., *"Hello! Please, could you..."* or *"Olá, por favor..."*).
* **Text Minification:** Flattens double line breaks and excessive trailing spaces.
* **Edge Cleanup:** Automatically cleans up orphan punctuation marks (`!`, `,`, `.`) left behind after trimming, and automatically capitalizes the final prompt.
* **Accurate Token Counting:** Uses the official `tiktoken` library (with the `cl100k_base` encoding template used by OpenAI's GPT-4o) to measure real-time savings.

---

## 📁 Project Structure

The repository is organized following containerization standards, making it ready for production deployment:

```text
├── src/
│   └── streamlit_app.py   # Main Application source code
├── .gitattributes         # Git configuration attributes
├── Dockerfile             # Production Docker container blueprint
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 🛠️ Local Installation & Setup

To run this project locally on your machine, follow these steps:

### Prerequisites

Make sure you have Python 3.10+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/luizcosta2191/tokens-compressor.git
cd tokens-compressor
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run src/streamlit_app.py
```

The app will automatically open in your default browser at:

```text
http://localhost:8501
```

---

## 🐳 Running with Docker

This project includes a lightweight Dockerfile based on `python:3.13.5-slim`.

To build and run the containerized version:

```bash
# Build the Docker image
docker build -t tokenshrink:latest .

# Run the container
docker run -p 8501:8501 tokenshrink:latest
```

Access the application via:

```text
http://localhost:8501
```

---

## 🌐 Cloud Deployment (Hugging Face Spaces)

This repository is fully compatible with Hugging Face Spaces Docker templates.

1. Create a new Space on Hugging Face using the Docker SDK.
2. Push this repository to your Hugging Face space remote.
3. The platform will automatically build the Dockerfile and host your live application.

---

## 🧪 Technologies Used

| Technology  | Purpose                                         |
| ----------- | ----------------------------------------------- |
| Python 3.13 | Core programming language                       |
| Streamlit   | Graphical user interface framework              |
| Tiktoken    | High-performance BPE tokenizer by OpenAI        |
| Docker      | Containerization and cloud deployment isolation |

---

## 📄 License

This project is open-source and available under the MIT License.
