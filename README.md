# Legal Document Analysis & Contract Review Agent

An AI-powered system for automated legal document analysis and contract review.

## Features

- Document parsing and analysis
- Risk identification and assessment
- Compliance checking
- Contract summarization
- Modification suggestions
- Integration capabilities with legal management systems

## Project Structure

```
legal_assistant/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── endpoints/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_analyzer.py
│   │   ├── risk_analyzer.py
│   │   └── compliance_checker.py
│   └── utils/
│       ├── __init__.py
│       └── text_processor.py
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py
├── requirements.txt
└── main.py
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Features Implementation

### 1. Document Analysis
- PDF and DOCX parsing
- Text extraction and preprocessing
- Entity recognition
- Key clause identification

### 2. Risk Assessment
- Risk level classification
- Potential issue highlighting
- Compliance violation detection
- Custom risk rules support

### 3. Contract Summarization
- Key points extraction
- Obligation identification
- Term and condition summarization
- Custom summary generation

### 4. Integration Capabilities
- REST API endpoints
- Webhook support
- Export functionality
- Custom integration adapters

## Development

### Running Tests
```bash
pytest
```

### Code Style
```bash
black .
isort .
flake8
``` 