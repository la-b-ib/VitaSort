import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re

# Page configuration for VitaSort
st.set_page_config(
    page_title="VitaSort - AI Resume Screening",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit app
st.title("🍎 VitaSort - AI Resume Screening & Ranking")
st.markdown("**Streamline your hiring process with intelligent AI-powered resume analysis**")

# Sidebar with VitaSort branding
with st.sidebar:
    st.markdown("### 🍎 VitaSort Control Panel")
    st.markdown("**Your AI-powered hiring assistant**")
    
    st.markdown("---")    
    # Instructions
    st.markdown("#### How to Use VitaSort")
    st.markdown("1. **Enter your job description**")
    st.markdown("2. **Upload candidate resumes (PDF)**")
    st.markdown("3. **Get instant AI rankings!**")
    st.markdown("4. **Explore advanced visualizations!**")
    
    st.markdown("---")
    st.markdown("**VitaSort v2.3**")
    st.markdown("**Developer: Labib Bin Shahed**")
    st.markdown("""
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/la-b-ib)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/la-b-ib/)
""")


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    # Scale similarity scores to range 1 to 100
    scores = (cosine_similarities * 100).round(2)
    return scores

# Advanced Visualization Functions
def create_parallel_coordinates(results, resumes, job_description):
    """Create a parallel coordinates chart for multi-dimensional analysis"""
    # Calculate multiple dimensions for each candidate
    parallel_data = []
    
    skill_categories = {
        'Technical': ['python', 'java', 'sql', 'programming', 'coding', 'software'],
        'Experience': ['years', 'experience', 'senior', 'lead', 'manager', 'director'],
        'Education': ['degree', 'university', 'college', 'bachelor', 'master', 'phd'],
        'Communication': ['communication', 'presentation', 'writing', 'team', 'collaboration']
    }
    
    for idx, (_, candidate) in enumerate(results.iterrows()):
        resume_text = resumes[idx].lower()
        
        # Calculate scores for each dimension
        dimensions = {
            'VitaSort_Score': candidate['🎯 VitaSort Score'],
            'Ranking': candidate['📈 Ranking']
        }
        
        for category, keywords in skill_categories.items():
            score = sum(resume_text.count(keyword) for keyword in keywords)
            dimensions[f'{category}_Score'] = min(score * 10, 100)
        
        dimensions['Resume'] = candidate['🍎 Resume'].replace('.pdf', '')
        parallel_data.append(dimensions)
    
    parallel_df = pd.DataFrame(parallel_data)
    
    # Create parallel coordinates plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=parallel_df['VitaSort_Score'],
                 colorscale='viridis',
                 showscale=True,
                 cmin=parallel_df['VitaSort_Score'].min(),
                 cmax=parallel_df['VitaSort_Score'].max()),
        dimensions=list([
            dict(range=[0, 100],
                 constraintrange=[0, 100],
                 label="VitaSort Score",
                 values=parallel_df['VitaSort_Score']),
            dict(range=[0, 100],
                 label="Technical Skills",
                 values=parallel_df['Technical_Score']),
            dict(range=[0, 100],
                 label="Experience Level",
                 values=parallel_df['Experience_Score']),
            dict(range=[0, 100],
                 label="Education",
                 values=parallel_df['Education_Score']),
            dict(range=[0, 100],
                 label="Communication",
                 values=parallel_df['Communication_Score']),
            dict(range=[1, len(results)],
                 label="Ranking",
                 values=parallel_df['Ranking'],
                 tickvals=list(range(1, len(results)+1)))
        ])
    ))
    
    fig.update_layout(
        title="📊 Parallel Coordinates Analysis",
        height=600,
        title_font=dict(size=18, color='#1E3A8A'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2E2E2E')
    )
    
    return fig

def create_radar_chart(results, resumes, job_description):
    """Create a radar chart showing multi-dimensional candidate analysis"""
    # Define skills categories for radar analysis
    skill_categories = {
        'Technical Skills': ['python', 'java', 'sql', 'programming', 'coding', 'software'],
        'Data Science': ['machine learning', 'data science', 'analytics', 'statistics', 'ai'],
        'Business Skills': ['management', 'leadership', 'strategy', 'business', 'project'],
        'Communication': ['communication', 'presentation', 'writing', 'collaboration', 'team'],
        'Experience': ['years', 'experience', 'senior', 'lead', 'manager', 'director']
    }
    
    # Analyze top 3 candidates
    top_candidates = results.head(3)
    
    fig = go.Figure()
    
    # Enhanced color palette for radar chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for idx, (_, candidate) in enumerate(top_candidates.iterrows()):
        resume_idx = results[results['🍎 Resume'] == candidate['🍎 Resume']].index[0]
        resume_text = resumes[resume_idx].lower()
        
        # Calculate scores for each category
        category_scores = []
        for category, keywords in skill_categories.items():
            score = sum(resume_text.count(keyword) for keyword in keywords)
            # Normalize to 0-100 scale
            normalized_score = min(score * 10, 100)
            category_scores.append(normalized_score)
        
        fig.add_trace(go.Scatterpolar(
            r=category_scores,
            theta=list(skill_categories.keys()),
            fill='toself',
            name=f"{candidate['🍎 Resume'].replace('.pdf', '')} (Score: {candidate['🎯 VitaSort Score']})",
            line=dict(color=colors[idx % len(colors)], width=3),
            fillcolor=f"rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.2)"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#E5E5E5',
                gridwidth=2,
                tickfont=dict(color='#666666')
            ),
            angularaxis=dict(
                tickfont=dict(color='#2E2E2E', size=12)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title=dict(
            text="🎯 Multi-Dimensional Candidate Analysis Radar Chart",
            font=dict(size=18, color='#1E3A8A')
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_word_cloud_comparison(job_description, top_resume_text):
    """Create word clouds for job description and top resume"""
    # Clean and prepare text
    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return ' '.join([word for word in text.split() if len(word) > 2])
    
    job_text_clean = clean_text(job_description)
    resume_text_clean = clean_text(top_resume_text)
    
    # Create word clouds with enhanced color schemes
    job_wordcloud = WordCloud(
        width=400, height=300, 
        background_color='white',
        colormap='plasma',  # Vibrant purple-pink gradient
        max_words=100,
        min_font_size=10
    ).generate(job_text_clean)
    
    resume_wordcloud = WordCloud(
        width=400, height=300, 
        background_color='white',
        colormap='viridis',  # Blue-green gradient
        max_words=100,
        min_font_size=10
    ).generate(resume_text_clean)
    
    return job_wordcloud, resume_wordcloud

def create_skills_heatmap(job_description, resumes, filenames):
    """Create a heatmap showing skill matching"""
    # Define common skills to look for
    skills = ['python', 'java', 'sql', 'machine learning', 'data science', 
              'analytics', 'project management', 'communication', 'leadership',
              'problem solving', 'teamwork', 'excel', 'powerbi', 'tableau']
    
    # Create matrix
    skill_matrix = []
    labels = ['Job Description'] + [f.replace('.pdf', '') for f in filenames]
    
    all_texts = [job_description.lower()] + [resume.lower() for resume in resumes]
    
    for text in all_texts:
        skill_scores = []
        for skill in skills:
            # Count occurrences and normalize
            count = text.count(skill.lower())
            skill_scores.append(min(count * 2, 10))  # Cap at 10 for better visualization
        skill_matrix.append(skill_scores)
    
    # Create heatmap with enhanced colors
    fig = px.imshow(skill_matrix, 
                    labels=dict(x="Skills", y="Documents", color="Match Score"),
                    x=[skill.title() for skill in skills],
                    y=labels,
                    color_continuous_scale="Turbo",  # Vibrant rainbow scale
                    title="🎯 Skills Matching Heatmap")
    
    # Enhanced styling
    fig.update_layout(
        height=600,
        title_font=dict(size=18, color='#1E3A8A'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2E2E2E')
    )
    
    return fig

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("📄 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process and rank resumes
if uploaded_files and job_description:
    st.header("🍎 VitaSort AI Analysis Results")
    st.markdown("**Smart ranking powered by advanced AI algorithms**")

    try:
        with st.spinner("🧠 VitaSort is analyzing your resumes..."):
            resumes = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes.append(text)

            # Rank resumes
            scores = rank_resumes(job_description, resumes)

        # Display results with VitaSort styling
        st.success("✅ Analysis complete! Here are your VitaSort rankings:")
        
        results = pd.DataFrame({
            "🍎 Resume": [file.name for file in uploaded_files], 
            "🎯 VitaSort Score": scores,
            "📈 Ranking": range(1, len(scores) + 1)
        }).sort_values(by="🎯 VitaSort Score", ascending=False)
        
        results["📈 Ranking"] = range(1, len(results) + 1)

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results Table", "🌟 Radar Analysis",
                                               "📊 Parallel Coordinates", "☁️ Word Clouds", 
                                               "🔥 Skills Heatmap"])
        
        with tab1:
            st.subheader("📋 Detailed Rankings Table")
            st.dataframe(results, use_container_width=True)
            
            # Show top candidate
            top_candidate = results.iloc[0]
            st.markdown(f"### 🏆 Top Match: {top_candidate['🍎 Resume']}")
            st.markdown(f"**VitaSort Score: {top_candidate['🎯 VitaSort Score']}/100**")
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Average Score", f"{scores.mean():.2f}")
            with col2:
                st.metric("🎯 Highest Score", f"{scores.max():.2f}")
            with col3:
                st.metric("📉 Lowest Score", f"{scores.min():.2f}")
            with col4:
                st.metric("📈 Score Range", f"{(scores.max() - scores.min()):.2f}")
        
        with tab2:
            st.subheader("🌟 Multi-Dimensional Radar Analysis")
            if len(results) >= 1:
                radar_chart = create_radar_chart(results, resumes, job_description)
                st.plotly_chart(radar_chart, use_container_width=True)
                
                st.markdown("#### � Radar Chart Analysis:")
                st.markdown("• **Technical Skills**: Programming languages and technical competencies")
                st.markdown("• **Data Science**: ML, AI, analytics, and statistical skills")
                st.markdown("• **Business Skills**: Management, leadership, and strategy experience")
                st.markdown("• **Communication**: Presentation, writing, and collaboration abilities")
                st.markdown("• **Experience**: Years of experience and seniority indicators")
                st.markdown("*Higher values indicate stronger presence of skills in that category*")
            else:
                st.info("Upload at least one resume to see radar analysis")
        
        with tab3:
            st.subheader("📊 Multi-Dimensional Parallel Coordinates")
            parallel_chart = create_parallel_coordinates(results, resumes, job_description)
            st.plotly_chart(parallel_chart, use_container_width=True)
            
            st.markdown("#### 📋 How to Use Parallel Coordinates:")
            st.markdown("• **Lines represent candidates** - each line is one resume")
            st.markdown("• **Vertical axes** show different skill dimensions")
            st.markdown("• **Line color** indicates VitaSort Score (darker = better match)")
            st.markdown("• **Drag on axes** to filter and explore patterns")
            st.markdown("• **Look for patterns** - parallel lines show similar profiles")

        with tab4:
            st.subheader("☁️ Word Cloud Analysis")
            if len(resumes) > 0:
                # Get top resume text
                top_resume_idx = results.index[0]
                top_resume_text = resumes[top_resume_idx]
                
                job_wc, resume_wc = create_word_cloud_comparison(job_description, top_resume_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 Job Description Keywords**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(job_wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("**🏆 Top Resume Keywords**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(resume_wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
        
        with tab5:
            st.subheader("🔥 Skills Matching Analysis")
            skills_heatmap = create_skills_heatmap(job_description, resumes, 
                                                  [file.name for file in uploaded_files])
            st.plotly_chart(skills_heatmap, use_container_width=True)
            
            st.markdown("#### 📋 How to Read This Heatmap:")
            st.markdown("• **Red/Orange**: Strong skill match")
            st.markdown("• **Yellow/Green**: Moderate skill match") 
            st.markdown("• **Blue/Purple**: Weak or no skill match")
            st.markdown("• **Higher scores** indicate more mentions of that skill")
        
    except Exception as e:
        st.error(f"🚨 VitaSort encountered an issue: {str(e)}")
        st.markdown("💡 **Troubleshooting Tips:**")
        st.markdown("• Ensure PDFs are readable and not password-protected")
        st.markdown("• Check that job description is provided")
        st.markdown("• Try uploading smaller files if processing fails")

# VitaSort footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🍎VitaSort v2.3</h3>
    <p><strong>AI-Powered Resume Screening & Ranking System</strong></p>
</div>
""", unsafe_allow_html=True)