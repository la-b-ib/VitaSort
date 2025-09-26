# 🍎 VitaSort: AI-Powered Resume Screening & Ranking System
Streamline your hiring process with VitaSort - an intelligent web-based application that leverages advanced AI to screen and rank resumes based on job descriptions, making your recruitment efficient and effective.

# 🖼️ Preview
![image](https://github.com/user-attachments/assets/1277d71c-9f92-4be0-accc-1f52c75efd9c)

# 📑 Project Document
For a detailed overview of VitaSort, you can refer to the project document:
[VitaSort - AI-Powered Resume Screening System - Project Document](AI-powered-Resume-Screening-and-Ranking-System.pdf)

# 🚀 Features
* **Intelligent Resume Screening**: Automatically analyze and screen resumes based on job requirements using advanced AI algorithms
* **Smart Candidate Ranking**: Rank candidates with AI-powered scoring for data-driven hiring decisions
* **PDF Processing**: Seamlessly upload and process resumes in PDF format
* **Real-time Analysis**: Get instant matching scores and candidate rankings
* **User-Friendly Interface**: Clean, intuitive design for streamlined recruitment workflow

# 🛠 Tech Stack

**Frontend**
* Streamlit (Interactive web application framework)

**Backend & AI**
* Python (Core application logic)
* scikit-learn (TF-IDF vectorization and cosine similarity)
* PyPDF2 (PDF text extraction)
* pandas (Data processing and analysis)

**Core Features**
* AI-powered resume analysis
* Real-time scoring algorithms
* Multi-file processing capabilities
* PyPDF2 (for extracting text from PDF files)
* scikit-learn (for text processing and similarity calculations)

# 📂 Directory Structure
```
github.com/codewithshek/AI-powered-Resume-Screening-and-Ranking-System/
├── Readme.md
├── AI-powered-Resume-Screening-and-Ranking-System-PPT.pdf
└── Main.py
 ```
# 📌 Setup & Installation

1. **Clone the Repository**
```bash
git clone https://github.com/codewithshek/AI-powered-Resume-Screening-and-Ranking-System.git
cd AI-powered-Resume-Screening-and-Ranking-System
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch VitaSort**
```bash
streamlit run Main.py
```

4. **Access VitaSort**
```
Open your browser and navigate to http://localhost:8501/ to start using VitaSort for intelligent resume screening.
```

# 📜 VitaSort Core Functions

* **extract_text_from_pdf(file)**: Intelligently extracts and processes text from PDF resumes
* **rank_resumes(job_description, resumes)**: Analyzes and ranks candidates using AI-powered cosine similarity algorithms for optimal job-candidate matching

# 💡 Future Enhancements
✅ Implement support for additional file formats (e.g., DOCX).

✅ Add advanced natural language processing (NLP) techniques for better resume analysis.

✅ Develop a mobile application for on-the-go resume screening and ranking.

# 🤝 Contributing
Feel free to fork and submit pull requests. Any contributions are welcome!

Made with ❤️ by D ABHISHEK YADAV as part of **AICTE- Internship on AI: Transformative Learning with TechSaksham – A joint CSR initiative of Microsoft & SAP, focusing on AI Technologies**
