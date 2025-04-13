# VitaSort - Resume Section Classifier & Career Ecosystem

VitaSort is an AI-powered resume section classifier and holistic career development platform. It transforms raw resumes into actionable insights, aligning skills with market demands, streamlining HR processes, and ensuring GDPR compliance. Built with a modular, cloud-native architecture, VitaSort combines NLP, machine learning, and real-time data integration to empower job seekers, organizations, and developers.

## Table of Contents
- [Overview](#overview)
- [Core Features](#core-features)
- [Implementation Details](#implementation-details)
- [Technical Specifications](#technical-specifications)
- [Impact & Use Cases](#impact--use-cases)
- [Differentiation & Innovation](#differentiation--innovation)
- [Roadmap](#roadmap)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
VitaSort redefines resume analysis by classifying sections (skills, education, experience, etc.) with precision and providing a suite of career tools. It parses multi-format resumes (PDF, DOCX, images), extracts entities using NLP, generates interactive visualizations, and aligns profiles with live job market trends. With enterprise-grade scalability, GDPR compliance, and collaborative features, VitaSort serves job seekers, HR teams, and developers building ATS integrations.

Developed by Labib Bin Shahed, a Computer Science and Engineering student at BRAC University. VitaSort reflects a passion for innovation at the intersection of AI, data science, and career growth.

## Core Features

### Resume Analysis & Enhancement
- **Multi-Format Parsing**: Supports PDF, DOCX, images (OCR), and web-based resumes.
- **NLP-Driven Entity Extraction**:
  - Detects skills, education, experience, and personal details.
  - Custom pattern matching for industry-specific terms (e.g., "blockchain", "Flutter").
- **Interactive Visualizations**:
  - Skill radar charts, experience timelines, education hierarchy trees.
- **Job Market Alignment**:
  - Compares resume skills with real-time market demands.
  - Identifies skill gaps and growth opportunities.

### Career Development Tools
- **Career Pathway Simulator**:
  - Predicts promotions, salary growth, and skill roadmaps using ML.
- **Interview Preparation Suite**:
  - Generates tailored technical and behavioral questions.
- **Automated Cover Letter Engine**:
  - Creates job-specific cover letters from resume data.

### Enterprise & Compliance
- **GDPR Compliance Check**:
  - PII detection, data retention audits, consent management.
- **Resume Comparison Tool**:
  - Semantic similarity scoring for candidate ranking.
- **Real-Time Collaboration Hub**:
  - Version control, comments, and live resume editing.

### Advanced Tools
- **Salary Benchmarking**:
  - Compares compensation with live market data.
- **Dynamic Resume Builder**:
  - Generates styled resumes (modern, classic templates).

## Implementation Details

### Architecture
- **Modular Design**: Independent modules for parsing, NLP, visualization, and compliance.
- **API-First Approach**: RESTful endpoints via FastAPI for HR system integration.
- **Async I/O**: Uses httpx for parallel data fetching (e.g., job trends).
- **Caching**: lru_cache optimizes NLP model loading and template reuse.

### Key Technologies
- **NLP Pipeline**: spaCy with custom rules and entity recognition.
- **Machine Learning**:
  - TF-IDF vectorization for resume comparison.
  - Cosine similarity for semantic analysis.
- **Visualization**: Plotly for interactive dashboards.
- **Cloud Integration**: AWS/GCP/Azure via CloudStorageAdapter.

### Security
- **PII Redaction**: Regex-based detection of emails, phones, SSNs.
- **GDPR Tools**: Automated audits for data retention and consent.
- **Role-Based Access**: JWT authentication for enterprise users.

## Technical Specifications

### Algorithms & Models
- **Text Vectorization**: TF-IDF with 500-dimensional embeddings.
- **Skill Matching**: Pattern-based Matcher with custom skill databases.
- **Career Simulation**: Linear regression for salary/promotion forecasting.

### Data Handling
- **Input Formats**: PDF (PyMuPDF), DOCX (python-docx), Images (Tesseract OCR).
- **Output Formats**: JSON, XML, HTML, Plotly graphs.
- **Storage**: Cloud-native via data lake integration.

### Performance
- **Parallel Processing**: ThreadPoolExecutor for batch resume analysis.
- **Model Optimization**: en_core_web_lg balances accuracy and speed.
- **Scalability**: Horizontal scaling via Kubernetes in cloud deployments.

## Impact & Use Cases

### For Job Seekers
- **Personalized Insights**: Identifies skill gaps and market alignment.
- **Time Savings**: 80% faster resume tailoring and cover letter generation.
- **Career Growth**: Data-driven pathways for promotions and transitions.

### For Organizations
- **HR Efficiency**: Automates resume screening and benchmarking.
- **Compliance**: Reduces GDPR violation risks with PII tools.
- **Talent Analytics**: Maps team skills and optimizes recruitment.

### For Developers
- **Extensibility**: Modular components for custom integrations.
- **API Ecosystem**: Supports plugins for LinkedIn, Indeed, or ATS systems.

## Differentiation & Innovation
1. **Live Market Integration**: Pulls real-time job trends via APIs.
2. **Collaborative Editing**: Google Docs-like resume collaboration.
3. **AI Feedback Loop**: Self-improving models through user interactions.
4. **Multi-Format Agnosticism**: Unified parsing for PDFs, images, and docs.
5. **Enterprise Scalability**: Cloud-native with SOC 2 compliance tools.

## Roadmap
- **Video Resume Analysis**: NLP for video/audio content.
- **Blockchain Certifications**: Credential verification via smart contracts, inspired by ICEIC 2025 research.
- **AR Interview Prep**: Virtual reality mock interviews.
- **Mental Fitness Scoring**: ML-based stress-testing for career paths.

## Installation

### Prerequisites
- Python 3.8+
- pip for package management
- Docker (optional for containerized deployment)
- Cloud credentials (AWS/GCP/Azure) for storage

### Requirements
```python
fastapi==0.95.0
spacy==3.5.0
plotly==5.14.0
pymupdf==1.21.0
python-docx==0.8.11
tesseract-ocr==5.3.0
httpx==0.23.0
```

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/la-b-ib/vitasort.git
   ``

2. Navigate to the project directory:
   ```bash
   cd vitasort
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Install system dependencies for Tesseract OCR
   sudo apt install tesseract-ocr  # For Debian/Ubuntu
   ```

4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Update `.env` with your API keys (e.g., AWS, job market APIs).

6. Run the application:
   ```bash
   python -m vitasort.app
   ```

7. (Optional) Deploy with Docker:
   ```bash
   docker-compose up --build
   ```

### Notes
- For Windows users, install Tesseract from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- Production deployment recommendation:
  ```bash
  docker-compose -f docker-compose.prod.yml up --build -d
  ```
- Ensure Python 3.8+ is installed

## Contributing
Contributions are welcome! To contribute to VitaSort 2.0, please follow these steps:

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   ```bash
   git commit -m "Add your feature description"
   ```
   ```bash
   git push origin feature/your-feature-name
   ```


## License
This project is licensed under the [MIT License](LICENSE).

## Contact
- **Author**: Labib Bin Shahed
- **Email**: [labib-x@protonmail.com](mailto:labib-x@protonmail.com)
- **GitHub**: [@la-b-ib](https://github.com/la-b-ib)
- **Portfolio**: [https://la-b-ib.github.io](https://la-b-ib.github.io)
