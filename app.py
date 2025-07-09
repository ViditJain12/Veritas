import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from datetime import datetime
from collections import Counter
import numpy as np
import os
from dotenv import load_dotenv
import validators
from openai import OpenAI
import pyperclip

# Load environment variables
load_dotenv()

# Initialize session state for history if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Instantiate OpenAI client (optional)
try:
    openai_api_key = os.getenv("OPENAI_API_KEY_2")
    if openai_api_key:
        client2 = OpenAI(api_key=openai_api_key)  # For Why Fake or Biased
    else:
        client2 = None
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    client2 = None

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def is_valid_url(url):
    """Check if the URL is valid and accessible."""
    try:
        # Check if URL is properly formatted
        if not validators.url(url):
            return False, "Invalid URL format. Please enter a valid URL starting with http:// or https://"
        
        # Check if URL is accessible
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=10)
        
        # Check for common error status codes
        if response.status_code == 403:
            return False, "This article is restricted and cannot be accessed. The website may require a subscription or login."
        elif response.status_code == 404:
            return False, "Article not found. The URL may be incorrect or the article may have been removed."
        elif response.status_code == 429:
            return False, "Too many requests. Please try again later."
        elif response.status_code >= 500:
            return False, "The website is currently experiencing issues. Please try again later."
        elif response.status_code != 200:
            return False, f"Unable to access the article (Error {response.status_code}). Please check the URL and try again."
        
        return True, "URL is valid and accessible"
    except requests.exceptions.Timeout:
        return False, "The request timed out. Please check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check your internet connection and try again."
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def extract_article_content(url):
    try:
        # Send request with headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        title = clean_text(title)
        
        # Try to find the main article content
        article_containers = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in str(x).lower() for term in ['article', 'content', 'post', 'story']))
        
        if article_containers:
            main_content = max(article_containers, key=lambda x: len(x.get_text()))
        else:
            main_content = soup.body
        
        # Remove unwanted elements
        for element in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract paragraphs
        paragraphs = main_content.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        text = clean_text(text)
        
        # Check if we got meaningful content
        if len(text.split()) < 50:  # Arbitrary threshold for minimum content
            return None, "Unable to extract meaningful content from the article. The article might be behind a paywall or require login."
        
        # Generate better summary
        sentences = sent_tokenize(text)
        if len(sentences) > 0:
            summary_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 5:
                    summary_sentences.append(sentence)
                if len(summary_sentences) >= 3:
                    break
            summary = ' '.join(summary_sentences)
        else:
            summary = "No summary available."
        
        # Extract keywords and their frequencies
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        word_freq = Counter(words)
        keywords = [word for word, _ in word_freq.most_common(10)]
        
        # Calculate sentiment for each sentence
        sentence_sentiments = []
        for sentence in sentences:
            if len(sentence.split()) > 5:
                sentiment = TextBlob(sentence).sentiment.polarity
                sentence_sentiments.append({
                    'sentence': sentence,
                    'sentiment': sentiment
                })
        
        # Format the full text into paragraphs
        formatted_text = '\n\n'.join([p.strip() for p in text.split('. ') if len(p.strip()) > 50])
        
        return {
            'title': title,
            'text': formatted_text,
            'summary': summary,
            'keywords': keywords,
            'word_freq': word_freq,
            'sentence_sentiments': sentence_sentiments,
            'url': url,
            'analyzed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, None
    except requests.exceptions.Timeout:
        return None, "The request timed out. Please check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check your internet connection and try again."
    except Exception as e:
        return None, f"Error extracting article content: {str(e)}"

def analyze_bias(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    emotional_words = ['shocking', 'outrageous', 'amazing', 'terrible', 'horrible', 'incredible']
    subjective_words = ['think', 'believe', 'feel', 'seem', 'appear']
    
    text_lower = text.lower()
    emotional_count = sum(1 for word in emotional_words if word in text_lower)
    subjective_count = sum(1 for word in subjective_words if word in text_lower)
    
    bias_score = min(100, (emotional_count * 10 + subjective_count * 5 + abs(sentiment) * 50))
    
    return bias_score

def analyze_fake_news(text):
    sensational_words = ['shocking', 'unbelievable', 'mind-blowing', 'you won\'t believe']
    text_lower = text.lower()
    sensational_count = sum(1 for word in sensational_words if word in text_lower)
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / len(text) if len(text) > 0 else 0
    
    fake_score = min(100, (
        sensational_count * 15 +
        exclamation_count * 2 +
        question_count * 2 +
        caps_ratio * 100
    ))
    
    return fake_score

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 33], 'color': "#2ecc71"},
                {'range': [33, 66], 'color': "#f1c40f"},
                {'range': [66, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=300,
        width=400,  # Fixed width
        margin=dict(l=20, r=20, t=50, b=20),  # Consistent margins
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        autosize=False  # Disable autosize
    )
    return fig

def create_word_frequency_chart(word_freq):
    # Get top 10 words
    top_words = dict(word_freq.most_common(10))
    
    fig = px.bar(
        x=list(top_words.keys()),
        y=list(top_words.values()),
        labels={'x': 'Words', 'y': 'Frequency'},
        title='Top 10 Most Frequent Words'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_sentiment_timeline(sentence_sentiments):
    # Create a timeline of sentiment
    sentiments = [s['sentiment'] for s in sentence_sentiments]
    sentences = [s['sentence'][:50] + '...' for s in sentence_sentiments]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=sentiments,
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#1f77b4'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Sentiment Analysis Timeline',
        xaxis_title='Sentence Position',
        yaxis_title='Sentiment Score',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig

def analyze_propaganda_techniques(text):
    """Analyze text for common propaganda techniques."""
    techniques = {
        'loaded_language': ['shocking', 'outrageous', 'amazing', 'terrible'],
        'fear_appeals': ['dangerous', 'threat', 'warning', 'crisis'],
        'bandwagon': ['everyone knows', 'nobody believes', 'everybody agrees'],
        'scare_tactics': ['if we don\'t act now', 'the consequences will be dire'],
        'glittering_generalities': ['freedom', 'democracy', 'justice', 'equality']
    }
    
    results = {}
    text_lower = text.lower()
    
    for technique, keywords in techniques.items():
        count = sum(1 for word in keywords if word in text_lower)
        results[technique] = count
    
    return results

def create_propaganda_chart(techniques):
    """Create a bar chart for propaganda techniques."""
    fig = px.bar(
        x=list(techniques.keys()),
        y=list(techniques.values()),
        labels={'x': 'Technique', 'y': 'Count'},
        title='Propaganda Techniques Analysis'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis={'tickangle': 45}
    )
    
    return fig

# Set up the Streamlit page
st.set_page_config(page_title="News Article Analyzer", layout="wide")
st.title("📰 News Article Analyzer")

# Add custom CSS for a cooler, scrollable history sidebar
st.markdown("""
    <style>
    .history-scroll-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 4px;
        margin-bottom: 1rem;
    }
    .history-card {
        background: linear-gradient(90deg, #23242b 60%, #23242b 100%);
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.10);
        margin-bottom: 14px;
        padding: 16px 18px;
        transition: box-shadow 0.2s, background 0.2s;
        border-left: 4px solid #ff4b4b;
        position: relative;
    }
    .history-card:hover {
        box-shadow: 0 6px 24px rgba(255,75,75,0.12);
        background: linear-gradient(90deg, #262730 60%, #2a2b32 100%);
    }
    .history-title {
        font-weight: 600;
        font-size: 1.05rem;
        color: #fff;
        margin-bottom: 2px;
    }
    .history-date {
        font-size: 0.92rem;
        color: #bbb;
        margin-bottom: 6px;
    }
    .history-meta {
        font-size: 0.93rem;
        color: #ff7b7b;
        margin-bottom: 2px;
    }
    /* Custom scrollbar */
    .history-scroll-container::-webkit-scrollbar {
        width: 7px;
        background: #23242b;
    }
    .history-scroll-container::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Add summary verdict at the top (after URL input)
def get_verdict(bias_score, fake_score):
    if fake_score > 66 or bias_score > 66:
        return ("❌ Potentially Fake or Highly Biased", "#e74c3c")
    elif fake_score > 33 or bias_score > 33:
        return ("⚠️ Needs Review", "#f1c40f")
    else:
        return ("✅ Likely Real", "#2ecc71")

# Add history sidebar
with st.sidebar:
    st.header("📚 Analysis History")
    st.markdown('<div class="history-scroll-container">', unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("🗑️ Clear History", key="clear_history_btn"):
            st.session_state.history = []
            st.experimental_rerun()
        for idx, entry in enumerate(reversed(st.session_state.history)):
            # Get color based on verdict
            if "❌" in entry.get('verdict', ''):
                border_color = "#e74c3c"  # Red for fake
                status_icon = "❌"
            elif "⚠️" in entry.get('verdict', ''):
                border_color = "#f1c40f"  # Yellow for needs review
                status_icon = "⚠️"
            else:
                border_color = "#2ecc71"  # Green for real
                status_icon = "✅"
            
            st.markdown(f'''
                <div class="history-card" style="border-left-color: {border_color};">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="flex: 1;">
                            <span style="font-size:1.2rem; margin-right:6px;">📰</span>
                            <div class="history-title">{entry['title']}</div>
                            <div class="history-date">{entry['date']}</div>
                            <div class="history-meta">Bias Score: {entry['bias_score']:.1f} | Fake News Score: {entry['fake_score']:.1f}</div>
                        </div>
                        <div style="font-size: 1.5rem; margin-left: 8px;" title="{entry.get('verdict', 'Unknown status')}">
                            {status_icon}
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            if st.button("Load Analysis", key=f"load_{idx}"):
                st.session_state.current_url = entry['url']
                st.experimental_rerun()
    else:
        st.info("No analysis history yet. Start by analyzing an article!")
    st.markdown('</div>', unsafe_allow_html=True)

# Custom CSS for dark mode compatibility and stylish tabs
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        min-width: 120px;
        white-space: pre-wrap;
        background: #23242b;
        border-radius: 24px 24px 0 0;
        color: #fff;
        font-size: 1.1rem;
        font-weight: 500;
        margin-right: 0px;
        transition: background 0.3s, color 0.3s, box-shadow 0.3s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: none;
        outline: none;
        padding: 0 28px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #31323a;
        color: #ff4b4b;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(255,75,75,0.10);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff7b7b 100%);
        color: #fff;
        font-weight: 700;
        box-shadow: 0 6px 24px rgba(255,75,75,0.15);
        border-radius: 24px 24px 0 0;
        border-bottom: 2px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# Input URL
url = st.text_input("Enter the URL of the news article:", st.session_state.get('current_url', ''))

if url:
    # Validate URL first
    is_valid, message = is_valid_url(url)
    if not is_valid:
        st.error(message)
    else:
        with st.spinner("Analyzing article..."):
            article_data, error_message = extract_article_content(url)
            
            if error_message:
                st.error(error_message)
            elif article_data:
                # Verdict
                bias_score = analyze_bias(article_data['text'])
                fake_score = analyze_fake_news(article_data['text'])
                verdict, verdict_color = get_verdict(bias_score, fake_score)
                
                # Add to history
                history_entry = {
                    'title': article_data['title'],
                    'url': url,
                    'date': article_data['analyzed_date'],
                    'bias_score': bias_score,
                    'fake_score': fake_score,
                    'verdict': verdict
                }
                
                # Check if this URL is already in history to avoid duplicates
                if not any(entry['url'] == url for entry in st.session_state.history):
                    st.session_state.history.append(history_entry)
                
                st.markdown(f"""
                <div style='margin-top: 10px; margin-bottom: 18px; padding: 16px 24px; border-radius: 12px; background: linear-gradient(90deg, {verdict_color} 0%, #23242b 100%); color: #fff; font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 12px;'>
                    <span>{verdict}</span>
                </div>
                """, unsafe_allow_html=True)
                # Copy Link button
                if st.button("🔗 Copy Analysis Link"):
                    pyperclip.copy(url)
                    st.success("Link copied to clipboard!")
                # Main header
                with st.container():
                    st.header(article_data['title'])
                    st.caption(f"Source: <a href='{article_data['url']}' target='_blank'>{article_data['url']}</a>", unsafe_allow_html=True)
                    st.caption(f"Analyzed on: {article_data['analyzed_date']}")
                # Gauges
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(bias_score, "Bias Score"), use_container_width=False)
                    st.markdown('<div style="margin-top: 8px; text-align:center;">🟢 <span style="color:#2ecc71;">Low</span> &nbsp; 🟡 <span style="color:#f1c40f;">Medium</span> &nbsp; 🔴 <span style="color:#e74c3c;">High</span></div>', unsafe_allow_html=True)
                    st.markdown('<div style="margin-top: 4px; color:#bbb; font-size:0.97rem;">Bias Score: <span title="How much the article uses emotional or subjective language">ℹ️</span></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
                    st.plotly_chart(create_gauge_chart(fake_score, "Fake News Score"), use_container_width=False)
                    st.markdown('<div style="margin-top: 8px; text-align:center;">🟢 <span style="color:#2ecc71;">Low</span> &nbsp; 🟡 <span style="color:#f1c40f;">Medium</span> &nbsp; 🔴 <span style="color:#e74c3c;">High</span></div>', unsafe_allow_html=True)
                    st.markdown('<div style="margin-top: 4px; color:#bbb; font-size:0.97rem;">Fake News Score: <span title="How much the article uses sensational or manipulative language">ℹ️</span></div>', unsafe_allow_html=True)
                # Divider
                st.markdown("<hr style='margin: 32px 0 24px 0; border: none; border-top: 2px solid #333;'>", unsafe_allow_html=True)
                # Tabs for secondary info
                st.markdown("<hr style='margin: 32px 0 24px 0; border: none; border-top: 2px solid #333;'>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["Summary", "Key Topics", "Analysis"])
                with tab1:
                    st.subheader("Article Summary")
                    st.markdown(f"""
                    <div style='background-color: #262730; padding: 20px; border-radius: 10px;'>
                        {article_data['summary']}
                    </div>
                    """, unsafe_allow_html=True)
                with tab2:
                    st.subheader("Key Topics")
                    keywords_html = " ".join([f"<span style='background-color: #1E1E1E; color: white; padding: 5px 10px; border-radius: 15px; margin: 5px; display: inline-block;'>{keyword}</span>" for keyword in article_data['keywords']])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    st.plotly_chart(create_word_frequency_chart(article_data['word_freq']), use_container_width=True)
                with tab3:
                    st.subheader("Sentiment Analysis")
                    st.plotly_chart(create_sentiment_timeline(article_data['sentence_sentiments']), use_container_width=True)
                    sentiments = [s['sentiment'] for s in article_data['sentence_sentiments']]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Sentiment", f"{np.mean(sentiments):.2f}")
                    with col2:
                        st.metric("Most Positive", f"{max(sentiments):.2f}")
                    with col3:
                        st.metric("Most Negative", f"{min(sentiments):.2f}")
                    st.subheader("Propaganda Techniques Analysis")
                    propaganda_techniques = analyze_propaganda_techniques(article_data['text'])
                    st.plotly_chart(create_propaganda_chart(propaganda_techniques), use_container_width=True)