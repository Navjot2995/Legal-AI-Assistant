import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import json
from datetime import datetime
from services.document_analyzer import DocumentAnalyzer
import requests

# Set page config
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize document analyzer
@st.cache_resource
def get_analyzer():
    return DocumentAnalyzer()

document_analyzer = get_analyzer()

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📄 Legal Document Analyzer")
st.markdown("""
    AI-powered legal document analysis and contract review system.
    Upload your legal documents to analyze risks, obligations, and key terms.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature:", ["Document Analysis", "Contract Comparison"])

# Add theme selector in sidebar
theme = st.sidebar.selectbox(
    "Choose Theme",
    ["Light", "Dark"],
    index=0
)

# Add export options in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Export Options")
export_format = st.sidebar.selectbox(
    "Export Format",
    ["PDF", "CSV", "JSON"],
    index=0
)

# Sidebar additions for Groq API
st.sidebar.markdown("---")
st.sidebar.subheader("Groq API Integration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key to use LLM features.")
groq_model = st.sidebar.selectbox(
    "Groq Model",
    ["llama2-70b-4096", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0
)

# Function to call Groq API

def call_groq_api(prompt, document_text, api_key, model):
    if not api_key:
        return "Please provide a Groq API key."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful legal document assistant."},
            {"role": "user", "content": f"{prompt}\n\nDocument:\n{document_text}"}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq API error: {str(e)}"

if page == "Document Analysis":
    st.header("Document Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a legal document (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Upload a PDF or DOCX file to analyze"
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                # Extract and analyze text
                text = document_analyzer.extract_text(temp_file_path)
                analysis = document_analyzer.analyze_document(text)

                # Document Statistics
                st.subheader("📊 Document Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", len(text.split()))
                with col2:
                    st.metric("Risk Score", f"{analysis['risk_score']:.1f}/10")
                with col3:
                    st.metric("Key Terms", len(analysis["key_terms"]))
                with col4:
                    st.metric("Named Entities", len(analysis["entities"]))

                # Main content in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Risks & Obligations", "Analysis", "Structure", "AI Assistant"])

                with tab1:
                    # Document Summary with expandable sections
                    with st.expander("📝 Document Summary", expanded=True):
                        st.write(analysis["summary"])

                    # Sentiment Analysis with enhanced visualization
                    st.subheader("📊 Sentiment Analysis")
                    sentiment_data = pd.DataFrame(analysis["sentiment"]["by_section"])
                    
                    # Create a more detailed sentiment visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=sentiment_data.index,
                        y=sentiment_data["score"],
                        name="Sentiment Score",
                        marker_color='rgb(55, 83, 109)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=sentiment_data.index,
                        y=sentiment_data["score"].rolling(window=3).mean(),
                        name="Trend",
                        line=dict(color='red', width=2)
                    ))
                    fig.update_layout(
                        title="Document Sentiment Analysis",
                        xaxis_title="Section",
                        yaxis_title="Sentiment Score",
                        hovermode="x unified",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    # Risks and Obligations in a grid
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚠️ Identified Risks")
                        # Create risk heatmap
                        risk_data = pd.DataFrame(analysis["risks"])
                        if not risk_data.empty:
                            risk_data["severity"] = pd.to_numeric(risk_data["severity"])
                            fig = px.treemap(
                                risk_data,
                                path=["severity"],
                                values="severity",
                                title="Risk Distribution by Severity"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            for risk in analysis["risks"]:
                                st.error(f"{risk['text']}\nSeverity: {risk['severity']}")

                    with col2:
                        st.subheader("📋 Key Obligations")
                        # Create timeline view of obligations
                        obligations_data = pd.DataFrame(analysis["obligations"])
                        if not obligations_data.empty:
                            fig = px.timeline(
                                obligations_data,
                                x_start="start_date",
                                x_end="end_date",
                                y="party",
                                title="Obligations Timeline"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            for obligation in analysis["obligations"]:
                                st.info(f"{obligation['text']}\nParty: {obligation['party']}")

                with tab3:
                    # Key Terms and Entities with enhanced visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔑 Key Terms")
                        terms_data = pd.DataFrame(analysis["key_terms"])
                        if not terms_data.empty:
                            fig = px.bar(
                                terms_data,
                                x="term",
                                y="importance",
                                title="Key Terms by Importance"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(terms_data)

                    with col2:
                        st.subheader("👥 Named Entities")
                        entities_data = pd.DataFrame(analysis["entities"])
                        if not entities_data.empty:
                            fig = px.pie(
                                entities_data,
                                names="type",
                                title="Entity Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(entities_data)

                with tab4:
                    # Document Structure Visualization
                    st.subheader("📑 Document Structure")
                    structure_data = pd.DataFrame(analysis["structure"])
                    if not structure_data.empty:
                        fig = px.treemap(
                            structure_data,
                            path=["section", "subsection"],
                            values="length",
                            title="Document Structure"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab5:
                    st.subheader("🤖 AI Assistant (Groq LLM)")
                    st.markdown("Ask questions about your document or request a summary using the selected Groq model.")

                    # Prompt templates for common legal tasks
                    prompt_templates = {
                        "Summarize this document": "Summarize this document.",
                        "List all risks": "List all risks mentioned in this document.",
                        "List all obligations": "List all obligations for each party in this document.",
                        "Extract all parties": "Identify all parties involved in this document.",
                        "Find key dates": "List all key dates and deadlines in this document.",
                        "Extract governing law": "What is the governing law specified in this document?",
                        "List termination conditions": "List all termination conditions in this document.",
                        "Find indemnification clauses": "Extract all indemnification clauses from this document.",
                        "Find confidentiality clauses": "Extract all confidentiality clauses from this document.",
                        "Custom": ""
                    }
                    template_choice = st.selectbox("Choose a prompt template", list(prompt_templates.keys()), index=0)

                    # Use session state to persist the prompt text
                    if 'user_prompt' not in st.session_state:
                        st.session_state['user_prompt'] = prompt_templates[template_choice]

                    # Update prompt if template changes
                    if st.session_state.get('last_template') != template_choice:
                        st.session_state['user_prompt'] = prompt_templates[template_choice]
                        st.session_state['last_template'] = template_choice

                    user_prompt = st.text_area("Enter your question or prompt", st.session_state['user_prompt'], key="ai_prompt")
                    if st.button("Ask Groq LLM"):
                        with st.spinner("Contacting Groq LLM..."):
                            groq_response = call_groq_api(user_prompt, text, groq_api_key, groq_model)
                            st.success(groq_response)

                # Export functionality
                if st.button("Export Analysis"):
                    export_data = {
                        "summary": analysis["summary"],
                        "risks": analysis["risks"],
                        "obligations": analysis["obligations"],
                        "key_terms": analysis["key_terms"],
                        "entities": analysis["entities"],
                        "sentiment": analysis["sentiment"],
                        "structure": analysis["structure"]
                    }
                    
                    if export_format == "JSON":
                        st.download_button(
                            "Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    elif export_format == "CSV":
                        # Convert relevant data to CSV
                        for key, value in export_data.items():
                            if isinstance(value, list):
                                df = pd.DataFrame(value)
                                st.download_button(
                                    f"Download {key}.csv",
                                    data=df.to_csv(index=False),
                                    file_name=f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

else:  # Contract Comparison
    st.header("Contract Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader(
            "Upload first document",
            type=["pdf", "docx"],
            key="file1"
        )
    
    with col2:
        file2 = st.file_uploader(
            "Upload second document",
            type=["pdf", "docx"],
            key="file2"
        )

    if file1 is not None and file2 is not None:
        if st.button("Compare Documents"):
            with st.spinner("Comparing documents..."):
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file1.name).suffix) as temp_file1:
                    temp_file1.write(file1.getvalue())
                    temp_file_path1 = temp_file1.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file2.name).suffix) as temp_file2:
                    temp_file2.write(file2.getvalue())
                    temp_file_path2 = temp_file2.name

                try:
                    # Extract text from both files
                    text1 = document_analyzer.extract_text(temp_file_path1)
                    text2 = document_analyzer.extract_text(temp_file_path2)

                    # Compare documents
                    comparison = document_analyzer.compare_contracts(text1, text2)

                    # Display results with enhanced visualizations
                    st.subheader("📊 Comparison Results")
                    
                    # Overall similarity with gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=comparison["similarity_score"] * 100,
                        title={'text': "Overall Similarity"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 75], 'color': "gray"},
                                   {'range': [75, 100], 'color': "darkgray"}
                               ]}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # Create tabs for different comparison aspects
                    tab1, tab2, tab3 = st.tabs(["Similar Clauses", "Different Clauses", "Missing Clauses"])

                    with tab1:
                        st.subheader("✅ Similar Clauses")
                        similar_data = pd.DataFrame(comparison["similarities"])
                        if not similar_data.empty:
                            fig = px.bar(
                                similar_data,
                                x="similarity",
                                title="Similarity Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            for item in comparison["similarities"]:
                                st.success(f"""
                                    Similarity: {item['similarity']*100:.1f}%
                                    Document 1: {item['clause1']}
                                    Document 2: {item['clause2']}
                                """)

                    with tab2:
                        st.subheader("⚠️ Different Clauses")
                        diff_data = pd.DataFrame(comparison["differences"])
                        if not diff_data.empty:
                            fig = px.scatter(
                                diff_data,
                                x="similarity",
                                y="length",
                                title="Difference Analysis"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            for item in comparison["differences"]:
                                st.warning(f"""
                                    Difference: {(1-item['similarity'])*100:.1f}%
                                    Document 1: {item['clause1']}
                                    Document 2: {item['clause2']}
                                """)

                    with tab3:
                        st.subheader("❌ Missing Clauses")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Missing in Document 2:")
                            missing2_data = pd.DataFrame(comparison["missing_clauses"]["missing_in_doc2"])
                            if not missing2_data.empty:
                                st.dataframe(missing2_data)
                            
                        with col2:
                            st.write("Missing in Document 1:")
                            missing1_data = pd.DataFrame(comparison["missing_clauses"]["missing_in_doc1"])
                            if not missing1_data.empty:
                                st.dataframe(missing1_data)

                    # Export comparison results
                    if st.button("Export Comparison"):
                        export_data = {
                            "similarity_score": comparison["similarity_score"],
                            "similarities": comparison["similarities"],
                            "differences": comparison["differences"],
                            "missing_clauses": comparison["missing_clauses"]
                        }
                        
                        if export_format == "JSON":
                            st.download_button(
                                "Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                except Exception as e:
                    st.error(f"An error occurred during comparison: {str(e)}")
                finally:
                    # Clean up temporary files
                    os.unlink(temp_file_path1)
                    os.unlink(temp_file_path2)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Legal Document Analyzer | Powered by AI</p>
    </div>
""", unsafe_allow_html=True) 