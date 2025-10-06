# Document Search Bot

Professional, minimal Document Q&A system built with Gradio and sentence-transformers.

This single-file Python app provides a simple Admin and User experience:

- Admin can upload or delete documents (PDF, Word, Excel, PowerPoint) up to 2 MB.
- User can ask questions and receive answers sourced from the uploaded documents.

Everything runs in-memory. The app extracts text from uploaded files, splits them into chunks, computes embeddings using a SentenceTransformers model, and performs nearest-neighbor search with FAISS.

## Features

- Simple role-based UI (Admin / User) via Gradio.
- Document parsing for: PDF (.pdf), Word (.docx), Excel (.xlsx/.xls), PowerPoint (.pptx).
- Chunking of documents for focused answers.
- Persistent embeddings saved to `embeddings.pkl` so uploaded data survives a restart.
- Minimal external dependencies and configuration.

## Project Structure

- `main.py` - The full application (Gradio UI + document processing + search) in a single file.
- `requirements.txt` - Exact dependency versions used for reproducibility.
- `uploaded_docs/` - Temporary folder where uploaded files are stored.
- `embeddings.pkl` - Serialized embeddings and texts (created automatically).

## Prerequisites

- Python 3.10+ (Python 3.13 tested in development).
- A machine with internet access (to download models on first run).

## Installation

1. Clone or copy this repository to your machine.
2. Change directory into the project root where `main.py` lives.
3. Create a virtual environment (recommended):

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

4. Install dependencies:

```bat
pip install -r requirements.txt
```

Note: The first run may download the `all-MiniLM-L6-v2` sentence-transformers model (a few hundred MB). Ensure you have enough disk space and a stable connection.

## Configuration

No API keys or credentials are required. The project uses the local `SentenceTransformer` model for embeddings; OpenAI is not used in this implementation.

Files are restricted to 2 MB each by default. Adjust `MAX_FILE_SIZE` in `main.py` if you need a different limit.

## Usage

Run the app:

```bat
python main.py
```

A Gradio web interface will open in your browser. Steps:

1. Choose role: `Admin` or `User`.
2. As Admin: upload documents (PDF/DOCX/XLSX/PPTX). The app will extract text, create embeddings, and persist them to `embeddings.pkl`.
3. As User: enter a question and press `Search`. The app returns the most relevant snippet and the source document name.

## Troubleshooting

- If Gradio shows UI errors about `.update()` on components, upgrade your Gradio version to match the `requirements.txt` or use the installed version that ships with this repo.
- If the sentence-transformers model download fails, retry with a stable connection or pre-download the model using `transformers` cache settings.
- If uploads silently fail, confirm file sizes are under 2 MB and the file types are supported.

## Security & Privacy

- Uploaded documents are stored locally in `uploaded_docs/` and embeddings in `embeddings.pkl`. Remove these files to delete data permanently.
- This project does not implement authentication; it is intended for local/demo use only.

## Extending the Project

- Replace local embeddings with OpenAI embeddings by swapping the `get_embedding` implementation.
- Add richer conversational responses by integrating an LLM for answer synthesis.
- Improve chunking and context windows for better precision.

## License

Provided as-is for educational and prototyping purposes. No license is attached—please adapt for your needs.
