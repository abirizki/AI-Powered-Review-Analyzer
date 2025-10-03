# advanced_dashboard.py - REDESIGN SIMPLE & PROFESSIONAL
import streamlit as st
import pandas as pd
from database import db
from nlp_processor import processor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import nltk
import ssl

# Auto-fix untuk NLTK resources
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 Downloading NLTK resources...")
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ NLTK resources downloaded!")

# 🎨 SIMPLE WHITE THEME CONFIGURATION
st.set_page_config(
    page_title="AI Review Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 MINIMAL CSS FOR CLEAN WHITE DESIGN
st.markdown("""
<style>
    /* Clean white background */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Simple sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Clean cards with subtle shadows */
    .custom-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* White tabs with clean borders */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #FFFFFF;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        border: 1px solid #dee2e6;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    
    /* Clean buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #dee2e6;
    }
    
    .stButton button:hover {
        border-color: #007bff;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        border-left: 4px solid #007bff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 🎨 CLEAN HEADER SECTION
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-size: 2.5em; margin-bottom: 10px;'>🚀 AI Review Analyzer Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d; font-size: 1.2em; margin-bottom: 30px;'>Advanced Sentiment Analysis & Customer Insights Platform</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Initialize session state
    if "current_review" not in st.session_state:
        st.session_state.current_review = "I love this product! It's amazing and works perfectly."
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "advanced_analysis" not in st.session_state:
        st.session_state.advanced_analysis = None
    
    # 🎨 CLEAN SIDEBAR DESIGN
    with st.sidebar:
        st.markdown("### 🎯 Analysis Settings")
        
        # Analysis Mode at the TOP as requested
        analysis_mode = st.selectbox(
            "Select Analysis Depth:",
            ["Standard", "Advanced", "Comprehensive"],
            index=1,
            key="analysis_mode_selector",
            help="Choose the depth of analysis for review processing"
        )
        st.session_state.analysis_mode = analysis_mode
        
        # Simple mode description
        mode_descriptions = {
            "Standard": "Basic sentiment analysis only",
            "Advanced": "Sentiment + aspect detection", 
            "Comprehensive": "Full analysis with features"
        }
        st.caption(f"*{mode_descriptions[analysis_mode]}*")
        
        st.markdown("---")
        
        # 🎨 SIMPLE CONTROL PANEL
        st.markdown("### 🔧 Actions")
        
        if st.button("🔄 Analyze All Reviews", use_container_width=True, type="primary"):
            with st.spinner("Analyzing all reviews..."):
                try:
                    count = db.analyze_sentiments()
                    st.success(f"✅ {count} reviews analyzed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("🤖 Train AI Model", use_container_width=True, type="secondary"):
            with st.spinner("Training AI model..."):
                try:
                    reviews = db.get_reviews(1000)
                    if len(reviews) >= 5:
                        success = processor.train_advanced_model(reviews)
                        if success:
                            st.success("✅ AI Model Trained!")
                            processor.load_advanced_model()
                            st.rerun()
                        else:
                            st.error("❌ Training failed!")
                    else:
                        st.warning(f"Need 5+ reviews (currently {len(reviews)})")
                except Exception as e:
                    st.error(f"Training error: {e}")
        
        st.markdown("---")
        
        # 🎨 CLEAN STATS SECTION
        st.markdown("### 📊 Quick Stats")
        
        try:
            stats = db.get_sentiment_stats()
            if not stats.empty:
                total_reviews = stats['count'].sum()
                
                st.metric("Total Reviews", total_reviews)
                
                # Sentiment distribution - clean and simple
                sentiment_data = {row['sentiment_label']: row['count'] for _, row in stats.iterrows()}
                
                for sentiment, count in sentiment_data.items():
                    percentage = (count/total_reviews*100) if total_reviews > 0 else 0
                    icon = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "⚠️"
                    st.write(f"{icon} {sentiment.title()}: {count} ({percentage:.1f}%)")
                    
            else:
                st.info("No data available")
        except Exception as e:
            st.error(f"Stats error: {e}")

    # 🎨 CLEAN MAIN DASHBOARD TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📈 Trends", "🏆 Products", "🤖 AI Insights", "🔍 Analysis"
    ])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_trends()
    
    with tab3:
        display_products()
    
    with tab4:
        display_ai_insights()
    
    with tab5:
        display_advanced_analysis()

def display_dashboard():
    """Clean dashboard dengan white theme"""
    st.subheader("Overview Dashboard")
    
    # 🎨 TOP METRICS ROW - SIMPLE DESIGN
    st.markdown("#### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            total_reviews = db.get_review_count()
            st.metric("Total Reviews", f"{total_reviews:,}")
        except:
            st.metric("Total Reviews", "0")
    
    with col2:
        try:
            stats = db.get_sentiment_stats()
            if not stats.empty:
                positive = stats[stats['sentiment_label'] == 'positive']['count'].sum()
                st.metric("Positive", f"{positive:,}")
            else:
                st.metric("Positive", "0")
        except:
            st.metric("Positive", "0")
    
    with col3:
        try:
            if not stats.empty:
                negative = stats[stats['sentiment_label'] == 'negative']['count'].sum()
                st.metric("Negative", f"{negative:,}")
            else:
                st.metric("Negative", "0")
        except:
            st.metric("Negative", "0")
    
    with col4:
        try:
            if not stats.empty:
                neutral = stats[stats['sentiment_label'] == 'neutral']['count'].sum()
                st.metric("Neutral", f"{neutral:,}")
            else:
                st.metric("Neutral", "0")
        except:
            st.metric("Neutral", "0")
    
    st.markdown("---")
    
    # 🎨 MAIN CONTENT AREA - CLEAN LAYOUT
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### Recent Reviews")
        
        try:
            reviews = db.get_reviews(6)
            if not reviews.empty:
                for i, row in reviews.iterrows():
                    with st.container():
                        sentiment = row.get('sentiment_label', 'Not analyzed')
                        confidence = row.get('confidence_score', 0) or 0
                        
                        # Simple sentiment display
                        if sentiment == "positive":
                            sentiment_color = "🟢"
                            badge_color = "success"
                        elif sentiment == "negative":
                            sentiment_color = "🔴"
                            badge_color = "error"
                        elif sentiment == "neutral":
                            sentiment_color = "🟡"
                            badge_color = "warning"
                        else:
                            sentiment_color = "⚪"
                            badge_color = "secondary"
                        
                        # Clean card design
                        with st.expander(f"{sentiment_color} {row.get('product_id', 'Unknown')} - {row.get('rating', 'N/A')}⭐", expanded=False):
                            st.write(f"**Review:** {str(row.get('review_text', 'No text'))[:100]}...")
                            st.write(f"**Sentiment:** {str(sentiment).title()} ({confidence:.0%} confidence)")
                            st.write(f"**Date:** {row.get('created_date', 'N/A')}")
            else:
                st.info("No reviews available. Add some reviews first!")
        except Exception as e:
            st.error(f"Error displaying reviews: {str(e)}")
    
    with col_right:
        st.markdown("#### Sentiment Distribution")
        
        try:
            stats = db.get_sentiment_stats()
            if not stats.empty:
                # Clean pie chart dengan light theme
                fig = px.pie(
                    stats, 
                    values='count', 
                    names='sentiment_label',
                    color='sentiment_label',
                    color_discrete_map={
                        'positive': '#28a745',
                        'negative': '#dc3545', 
                        'neutral': '#ffc107'
                    },
                    hole=0.4
                )
                fig.update_layout(
                    title="",
                    height=300,
                    showlegend=True,
                    font=dict(color='black'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No sentiment data available")
        except Exception as e:
            st.error(f"Error loading chart: {str(e)}")

def display_trends():
    """Clean trends analysis dengan white theme"""
    st.subheader("Trend Analysis")
    
    # 🎨 SIMPLE CONTROLS
    col_controls1, col_controls2 = st.columns(2)
    
    with col_controls1:
        days = st.selectbox(
            "Analysis Period:",
            [7, 30, 60, 90],
            index=1
        )
    
    with col_controls2:
        granularity = st.selectbox(
            "View By:",
            ["Daily", "Weekly", "Monthly"],
            index=0
        )
    
    try:
        trends = db.get_sentiment_trends(days)
        
        if not trends.empty and len(trends) > 1:
            # Calculate percentages
            trends = trends.copy()
            trends['positive_percentage'] = (trends['positive_count'] / trends['total_reviews'] * 100).round(2)
            trends['negative_percentage'] = (trends['negative_count'] / trends['total_reviews'] * 100).round(2)
            trends['neutral_percentage'] = (trends['neutral_count'] / trends['total_reviews'] * 100).round(2)
            
            # Convert granularity
            if granularity == "Weekly":
                trends['period'] = pd.to_datetime(trends['date']).dt.to_period('W').dt.start_time
                trends = trends.groupby('period').agg({
                    'positive_percentage': 'mean',
                    'negative_percentage': 'mean',
                    'neutral_percentage': 'mean',
                    'total_reviews': 'sum'
                }).reset_index()
                x_axis = 'period'
            elif granularity == "Monthly":
                trends['period'] = pd.to_datetime(trends['date']).dt.to_period('M').dt.start_time
                trends = trends.groupby('period').agg({
                    'positive_percentage': 'mean',
                    'negative_percentage': 'mean',
                    'neutral_percentage': 'mean',
                    'total_reviews': 'sum'
                }).reset_index()
                x_axis = 'period'
            else:
                trends['period'] = pd.to_datetime(trends['date'])
                x_axis = 'period'
            
            # 🎨 CLEAN TREND CHART
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trends[x_axis], 
                y=trends['positive_percentage'],
                mode='lines+markers',
                name='Positive %',
                line=dict(color='#28a745', width=3),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=trends[x_axis], 
                y=trends['negative_percentage'],
                mode='lines+markers', 
                name='Negative %',
                line=dict(color='#dc3545', width=3),
                marker=dict(size=6)
            ))
            
            fig.add_trace(go.Scatter(
                x=trends[x_axis], 
                y=trends['neutral_percentage'],
                mode='lines+markers', 
                name='Neutral %',
                line=dict(color='#ffc107', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Sentiment Trends - {granularity} View",
                xaxis_title="Time Period",
                yaxis_title="Percentage (%)",
                height=400,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 🎨 SIMPLE TREND METRICS
            st.subheader("Trend Summary")
            
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                positive_trend = trends['positive_percentage'].iloc[-1] - trends['positive_percentage'].iloc[0]
                st.metric(
                    "Positive Trend", 
                    f"{trends['positive_percentage'].iloc[-1]:.1f}%",
                    f"{positive_trend:+.1f}%"
                )
            
            with col_met2:
                negative_trend = trends['negative_percentage'].iloc[-1] - trends['negative_percentage'].iloc[0]
                st.metric(
                    "Negative Trend",
                    f"{trends['negative_percentage'].iloc[-1]:.1f}%", 
                    f"{negative_trend:+.1f}%"
                )
            
            with col_met3:
                total_reviews = trends['total_reviews'].sum()
                st.metric("Total Reviews", f"{total_reviews:,}")
            
        else:
            st.info("Not enough trend data available. Analyze more reviews to see trends.")
            
    except Exception as e:
        st.error(f"Error loading trends: {str(e)}")

def display_products():
    """Clean product analysis dengan white theme"""
    st.subheader("Product Performance")
    
    try:
        product_stats = db.get_product_stats()
        
        if not product_stats.empty:
            # 🎨 CLEAN PRODUCT TABLE
            st.markdown("#### Product Ranking")
            
            display_df = product_stats.copy()
            display_df['positive_ratio'] = (display_df['positive_count'] / display_df['total_reviews'] * 100).round(1)
            display_df = display_df.sort_values('positive_ratio', ascending=False)
            
            # Simple table view
            st.dataframe(
                display_df[['product_id', 'total_reviews', 'avg_rating', 'positive_ratio']].head(10),
                use_container_width=True,
                column_config={
                    "product_id": "Product ID",
                    "total_reviews": "Reviews",
                    "avg_rating": "Avg Rating",
                    "positive_ratio": "Positive %"
                }
            )
            
            # 🎨 SIMPLE PRODUCT COMPARISON
            st.markdown("#### Top Products Comparison")
            
            top_products = product_stats.nlargest(4, 'total_reviews')
            
            fig = go.Figure()
            
            colors = ['#28a745', '#17a2b8', '#ffc107', '#dc3545']
            
            for i, (_, product) in enumerate(top_products.iterrows()):
                fig.add_trace(go.Bar(
                    name=product['product_id'][:12] + ("..." if len(product['product_id']) > 12 else ""),
                    x=['Positive', 'Neutral', 'Negative'],
                    y=[product['positive_count'], product['neutral_count'], product['negative_count']],
                    marker_color=colors[i % len(colors)]
                ))
            
            fig.update_layout(
                title="Sentiment Distribution - Top Products",
                barmode='group',
                height=400,
                xaxis_title="Sentiment",
                yaxis_title="Number of Reviews",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='black')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No product data available. Analyze some reviews to see product performance.")
    except Exception as e:
        st.error(f"Error loading product stats: {str(e)}")

def display_ai_insights():
    """Clean AI insights dengan white theme"""
    st.subheader("AI Model Insights")
    
    # Auto-try to load model
    try:
        if not hasattr(processor, 'advanced_model') or processor.advanced_model is None:
            processor.load_advanced_model()
    except:
        pass
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Model Performance")
        
        try:
            model_loaded = (hasattr(processor, 'advanced_model') and 
                          processor.advanced_model is not None)
            
            metrics_available = (hasattr(processor, 'performance_metrics') and 
                               processor.performance_metrics and 
                               len(processor.performance_metrics) > 0)
            
            if model_loaded and metrics_available:
                metrics = processor.performance_metrics
                
                # 🎨 CLEAN PERFORMANCE CARDS
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    accuracy = metrics.get('accuracy', 0)
                    st.metric("Accuracy", f"{accuracy:.1%}")
                
                with col_met2:
                    cv_score = metrics.get('cv_mean', 0)
                    st.metric("CV Score", f"{cv_score:.1%}")
                
                with col_met3:
                    samples = metrics.get('training_samples', 0)
                    st.metric("Training Samples", f"{samples:,}")
                
                # Training info
                st.markdown("#### Training Information")
                training_date = metrics.get('training_date', 'Unknown')
                st.write(f"**Last Trained:** {training_date}")
                
                if hasattr(processor.advanced_model, 'n_estimators'):
                    st.write(f"**Model Type:** Random Forest ({processor.advanced_model.n_estimators} trees)")
                
                # Feature importance
                if 'feature_importance' in metrics and metrics['feature_importance']:
                    st.markdown("#### Feature Importance")
                    
                    feature_importance = metrics['feature_importance']
                    features = list(feature_importance.keys())
                    importance_values = list(feature_importance.values())
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance_values
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='black')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                # 🎨 SIMPLE SETUP GUIDE
                st.info("AI Model Not Yet Trained")
                
                st.markdown("""
                To enable AI insights:
                1. Ensure you have 5+ reviews
                2. Click 'Train AI Model' in the sidebar
                3. Wait for training to complete
                """)
                
                # Data status
                try:
                    review_count = db.get_review_count()
                    if review_count >= 5:
                        st.success(f"✅ Ready to train: {review_count} reviews available")
                    else:
                        st.warning(f"Need more data: {review_count}/5 reviews")
                except:
                    pass
                
        except Exception as e:
            st.error(f"Error loading AI insights: {str(e)}")
    
    with col2:
        st.markdown("#### Model Status")
        
        # 🎨 CLEAN STATUS INDICATORS
        model_loaded = hasattr(processor, 'advanced_model') and processor.advanced_model is not None
        if model_loaded:
            st.success("✅ Advanced Model")
        else:
            st.info("ℹ️ Basic Model")
        
        vectorizer_loaded = hasattr(processor, 'vectorizer') and processor.vectorizer is not None
        if vectorizer_loaded:
            st.success("✅ Text Vectorizer")
        else:
            st.info("ℹ️ No Vectorizer")
        
        try:
            review_count = db.get_review_count()
            if review_count >= 5:
                st.success(f"✅ {review_count} Reviews")
            else:
                st.warning(f"⚠️ {review_count} Reviews")
        except:
            st.error("❌ Data Error")
        
        st.markdown("---")
        st.markdown("#### Quick Actions")
        
        if st.button("Train Model", use_container_width=True):
            with st.spinner("Training..."):
                try:
                    reviews = db.get_reviews(1000)
                    if len(reviews) >= 5:
                        success = processor.train_advanced_model(reviews)
                        if success:
                            st.success("Model trained!")
                            st.rerun()
                    else:
                        st.error("Need 5+ reviews")
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
        
        if st.button("Load Model", use_container_width=True):
            with st.spinner("Loading..."):
                if processor.load_advanced_model():
                    st.success("Model loaded!")
                    st.rerun()
                else:
                    st.info("No saved model")

def display_advanced_analysis():
    """Clean analysis interface dengan white theme"""
    st.subheader("Review Analysis")
    
    # 🎨 CLEAN TWO-COLUMN LAYOUT
    col_input, col_preview = st.columns([2, 1])
    
    with col_input:
        st.markdown("#### Analyze Review Text")
        
        review_text = st.text_area(
            "Enter review text:",
            value=st.session_state.current_review,
            height=120,
            placeholder="Paste your review here for analysis..."
        )
        
        st.session_state.current_review = review_text
        
        # Analysis settings
        analysis_mode = st.session_state.get('analysis_mode', 'Advanced')
        st.write(f"**Analysis Mode:** {analysis_mode}")
        
        if st.button("Analyze Review", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        if analysis_mode in ["Advanced", "Comprehensive"]:
                            analysis = processor.analyze_review_comprehensive(review_text)
                            st.session_state.analysis_result = analysis
                        else:
                            sentiment, confidence = processor.predict_sentiment_advanced(review_text)
                            st.session_state.analysis_result = {
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'aspects': [],
                                'text_length': len(review_text),
                                'word_count': len(review_text.split())
                            }
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
            else:
                st.warning("Please enter review text")
    
    with col_preview:
        st.markdown("#### Quick Preview")
        
        if review_text.strip():
            try:
                sentiment, confidence = processor.predict_sentiment_advanced(review_text)
                
                if sentiment == "positive":
                    st.success(f"**{sentiment.title()}**")
                elif sentiment == "negative":
                    st.error(f"**{sentiment.title()}**")
                else:
                    st.warning(f"**{sentiment.title()}**")
                
                st.metric("Confidence", f"{confidence:.0%}")
                
                # Quick stats
                word_count = len(review_text.split())
                st.write(f"**Words:** {word_count}")
                
                # Aspect preview
                if analysis_mode in ["Advanced", "Comprehensive"]:
                    aspects = processor.detect_aspects(review_text)
                    if aspects:
                        st.write("**Aspects:**")
                        for aspect in aspects:
                            st.write(f"- {aspect.title()}")
                
            except Exception as e:
                st.error(f"Preview error: {e}")
        else:
            st.info("Enter text to see preview")
    
    # 🎨 CLEAN RESULTS DISPLAY
    if st.session_state.analysis_result:
        st.markdown("---")
        st.markdown("#### Analysis Results")
        
        analysis = st.session_state.analysis_result
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        with col_res1:
            st.metric("Sentiment", analysis['sentiment'].title())
        with col_res2:
            st.metric("Confidence", f"{analysis['confidence']:.0%}")
        with col_res3:
            st.metric("Text Length", analysis['text_length'])
        with col_res4:
            st.metric("Word Count", analysis['word_count'])
        
        if analysis['aspects']:
            st.markdown("**Detected Aspects:**")
            for aspect in analysis['aspects']:
                st.write(f"• {aspect.title()}")

if __name__ == "__main__":
    main()