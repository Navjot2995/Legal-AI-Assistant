# Legal Document Analyzer

An AI-powered legal document analysis and contract review system built with Streamlit. This app allows users to upload legal documents (PDF or DOCX), analyze them for risks, obligations, key terms, and more, and compare two contracts for similarities and differences. It also integrates with Groq's LLM API for advanced document Q&A and summarization.

## Features

- **Document Analysis:**
  - Upload PDF or DOCX files for analysis.
  - View document summary, identified risks, key obligations, and sentiment analysis.
  - Interactive visualizations for document insights.
  - Export analysis results in JSON or CSV format.

- **Contract Comparison:**
  - Upload two documents to compare similarities and differences.
  - View overall similarity score, similar clauses, different clauses, and missing clauses.
  - Interactive visualizations for comparison results.

- **AI Assistant (Groq LLM):**
  - Enter your Groq API key and select a model (Llama 2, Mixtral, Gemma, etc.).
  - Use prompt templates for common legal tasks (summarization, risk extraction, etc.).
  - Ask custom questions about your uploaded document.

## Setup

### Prerequisites

- Python 3.8 or later
- A Groq API key (for LLM features)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/legal-document-analyzer.git
   cd legal-document-analyzer
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

### Running the App

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Document Analysis:**
   - Upload a legal document (PDF or DOCX).
   - View the analysis results in the Overview, Risks & Obligations, Analysis, and Structure tabs.
   - Use the AI Assistant tab to ask questions or request summaries using Groq LLM.

2. **Contract Comparison:**
   - Upload two documents to compare.
   - View the comparison results, including similarity scores and clause differences.

3. **AI Assistant:**
   - Enter your Groq API key in the sidebar.
   - Select a prompt template or enter a custom prompt.
   - Click "Ask Groq LLM" to get a response.

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Connect your GitHub repo and set the main file as `app.py`.
4. Add your Groq API key as a Secret in the app settings.
5. Deploy the app.

### Render

1. Push your code to GitHub.
2. Go to [Render.com](https://render.com/).
3. Create a new Web Service and connect your GitHub repo.
4. Set the build command: `pip install -r requirements.txt`.
5. Set the start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
6. Add your Groq API key as an environment variable.
7. Deploy the app.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/).
- Powered by [Groq](https://groq.com/) for LLM features.
- Uses [spaCy](https://spacy.io/) and [transformers](https://huggingface.co/transformers/) for document analysis. 