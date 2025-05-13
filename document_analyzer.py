from typing import Dict, List, Optional, Tuple
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import PyPDF2
from docx import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

class DocumentAnalyzer:
    def __init__(self):
        # Load spaCy model for legal text
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize transformers pipeline for text classification
        self.classifier = pipeline(
            "text-classification",
            model="nlpaueb/legal-bert-base-uncased"
        )
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load clause classification model
        self.clause_classifier = pipeline(
            "text-classification",
            model="nlpaueb/legal-bert-base-uncased"
        )
        
        # Common legal terms and their categories
        self.legal_terms = {
            "liability": "risk",
            "indemnification": "risk",
            "warranty": "obligation",
            "termination": "condition",
            "confidentiality": "obligation",
            "intellectual property": "rights",
            "governing law": "jurisdiction",
            "dispute resolution": "procedure"
        }
        
        # Clause categories
        self.clause_categories = [
            "definitions",
            "obligations",
            "rights",
            "termination",
            "confidentiality",
            "intellectual_property",
            "indemnification",
            "limitation_of_liability",
            "governing_law",
            "dispute_resolution"
        ]

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or DOCX files."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def analyze_document(self, text: str) -> Dict:
        """Analyze document content and return insights."""
        doc = self.nlp(text)
        
        # Extract key information
        analysis = {
            "summary": self._generate_summary(doc),
            "risks": self._identify_risks(doc),
            "obligations": self._identify_obligations(doc),
            "key_terms": self._extract_key_terms(doc),
            "entities": self._extract_entities(doc),
            "clauses": self._classify_clauses(doc),
            "sentiment": self._analyze_sentiment(doc)
        }
        
        return analysis

    def compare_contracts(self, text1: str, text2: str) -> Dict:
        """Compare two contracts and identify differences and similarities."""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Extract clauses from both documents
        clauses1 = self._extract_clauses(doc1)
        clauses2 = self._extract_clauses(doc2)
        
        # Compare clauses
        comparison = {
            "similarities": self._find_similar_clauses(clauses1, clauses2),
            "differences": self._find_different_clauses(clauses1, clauses2),
            "missing_clauses": self._find_missing_clauses(clauses1, clauses2),
            "similarity_score": self._calculate_document_similarity(text1, text2)
        }
        
        return comparison

    def _extract_clauses(self, doc) -> List[Dict]:
        """Extract and classify clauses from the document."""
        clauses = []
        current_clause = []
        
        for sent in doc.sents:
            # Check if this is a new clause (e.g., starts with a number or specific keywords)
            if re.match(r'^\d+\.|^[A-Z]\.|^\([a-z]\)', sent.text.strip()):
                if current_clause:
                    clauses.append({
                        "text": " ".join(current_clause),
                        "classification": self._classify_clause(" ".join(current_clause))
                    })
                current_clause = [sent.text]
            else:
                current_clause.append(sent.text)
        
        # Add the last clause
        if current_clause:
            clauses.append({
                "text": " ".join(current_clause),
                "classification": self._classify_clause(" ".join(current_clause))
            })
        
        return clauses

    def _classify_clause(self, text: str) -> str:
        """Classify a clause into one of the predefined categories."""
        # Use the clause classifier to determine the category
        result = self.clause_classifier(text)
        return result[0]['label']

    def _find_similar_clauses(self, clauses1: List[Dict], clauses2: List[Dict]) -> List[Dict]:
        """Find similar clauses between two documents."""
        similar_clauses = []
        
        for c1 in clauses1:
            for c2 in clauses2:
                similarity = self._calculate_similarity(c1["text"], c2["text"])
                if similarity > 0.8:  # Threshold for similarity
                    similar_clauses.append({
                        "clause1": c1["text"],
                        "clause2": c2["text"],
                        "similarity": similarity,
                        "classification": c1["classification"]
                    })
        
        return similar_clauses

    def _find_different_clauses(self, clauses1: List[Dict], clauses2: List[Dict]) -> List[Dict]:
        """Find clauses that are different between two documents."""
        different_clauses = []
        
        for c1 in clauses1:
            for c2 in clauses2:
                if c1["classification"] == c2["classification"]:
                    similarity = self._calculate_similarity(c1["text"], c2["text"])
                    if similarity < 0.8:  # Threshold for difference
                        different_clauses.append({
                            "clause1": c1["text"],
                            "clause2": c2["text"],
                            "similarity": similarity,
                            "classification": c1["classification"]
                        })
        
        return different_clauses

    def _find_missing_clauses(self, clauses1: List[Dict], clauses2: List[Dict]) -> Dict:
        """Find clauses that are present in one document but missing in the other."""
        categories1 = set(c["classification"] for c in clauses1)
        categories2 = set(c["classification"] for c in clauses2)
        
        return {
            "missing_in_doc2": list(categories1 - categories2),
            "missing_in_doc1": list(categories2 - categories1)
        }

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embedding1 = self.sentence_transformer.encode([text1])[0]
        embedding2 = self.sentence_transformer.encode([text2])[0]
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def _calculate_document_similarity(self, text1: str, text2: str) -> float:
        """Calculate overall similarity between two documents."""
        # Split documents into sentences
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        sentences1 = [sent.text for sent in doc1.sents]
        sentences2 = [sent.text for sent in doc2.sents]
        
        # Calculate embeddings for all sentences
        embeddings1 = self.sentence_transformer.encode(sentences1)
        embeddings2 = self.sentence_transformer.encode(sentences2)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        # Calculate average similarity
        return float(np.mean(similarity_matrix))

    def _analyze_sentiment(self, doc) -> Dict:
        """Analyze the sentiment of different parts of the document."""
        sentiments = {
            "overall": 0,
            "by_section": []
        }
        
        for sent in doc.sents:
            # Use the classifier to determine sentiment
            result = self.classifier(sent.text)
            sentiment_score = float(result[0]['score'])
            
            sentiments["by_section"].append({
                "text": sent.text,
                "sentiment": result[0]['label'],
                "score": sentiment_score
            })
        
        # Calculate overall sentiment
        sentiments["overall"] = np.mean([s["score"] for s in sentiments["by_section"]])
        
        return sentiments

    def _generate_summary(self, doc) -> str:
        """Generate a summary of the document."""
        # Implement summarization logic
        sentences = [sent.text for sent in doc.sents]
        # For now, return first few sentences as summary
        return " ".join(sentences[:3])

    def _identify_risks(self, doc) -> List[Dict]:
        """Identify potential risks in the document."""
        risks = []
        for sent in doc.sents:
            # Check for risk-related terms
            if any(term in sent.text.lower() for term in ["risk", "liability", "indemnification"]):
                risks.append({
                    "text": sent.text,
                    "severity": self._calculate_risk_severity(sent)
                })
        return risks

    def _identify_obligations(self, doc) -> List[Dict]:
        """Identify obligations in the document."""
        obligations = []
        for sent in doc.sents:
            # Check for obligation-related terms
            if any(term in sent.text.lower() for term in ["shall", "must", "will", "agree to"]):
                obligations.append({
                    "text": sent.text,
                    "party": self._identify_party(sent)
                })
        return obligations

    def _extract_key_terms(self, doc) -> List[Dict]:
        """Extract key terms from the document."""
        key_terms = []
        for term, category in self.legal_terms.items():
            if term in doc.text.lower():
                key_terms.append({
                    "term": term,
                    "category": category,
                    "importance": self._calculate_term_importance(term, doc)
                })
        return key_terms

    def _extract_entities(self, doc) -> List[Dict]:
        """Extract named entities from the document."""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

    def _identify_party(self, sent) -> str:
        """Identify the party responsible for an obligation."""
        # Simple implementation - can be enhanced
        for ent in sent.ents:
            if ent.label_ in ["ORG", "PERSON"]:
                return ent.text
        return "Unknown"

    def _calculate_risk_severity(self, sent) -> float:
        """Calculate the severity of a risk."""
        # Simple implementation - can be enhanced
        severity_terms = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        for term, score in severity_terms.items():
            if term in sent.text.lower():
                return score
        return 0.5  # Default to medium severity

    def _calculate_term_importance(self, term: str, doc) -> float:
        """Calculate the importance of a term in the document."""
        # Simple implementation - can be enhanced
        term_count = doc.text.lower().count(term)
        return min(term_count / 10, 1.0)  # Normalize to 0-1 range 