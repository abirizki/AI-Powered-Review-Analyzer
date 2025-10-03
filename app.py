# Buat advanced_dashboard.py dengan semua fitur baru
$advanced_dashboard = @'
import streamlit as st
import pandas as pd
from database import db
from nlp_processor import processor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="AI Review Analyzer Pro+",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 AI-Powered Customer Review Analyzer Pro+")
    st.markdown("### Advanced Analytics with Enhanced AI Models")
    
    # Initialize session state
    if "current_review" not in st.session_state:
        st.session_state.current_review = "I love this product! It's amazing and works perfectly."
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "advanced_analysis" not in st.session_state:
        st.session_state.advanced_analysis = None
    
    # Sidebar dengan advanced controls
    with st.sidebar:
        st.header("🔧 Advanced Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Analyze All", type="primary", use_container_width=True):
                with st.spinner("Comprehensive analysis..."):
                    try:
                        count = db.analyze_sentiments()
                        st.success(f"✅ Analyzed {count} reviews!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("🤖 Train AI", type="secondary", use_container_width=True):
                with st.spinner("Training advanced AI model..."):
                    try:
                        reviews = db.get_reviews(1000)
                        success = processor.train_advanced_model(reviews)
                        if success:
                            st.success("✅ AI Model Trained!")
                        else:
                            st.info("ℹ️ Using rule-based analysis")
                    except Exception as e:
                        st.error(f"Training error: {e}")
        
        st.markdown("---")
        st.markdown("### 📈 Real-time Stats")
        
        try:
            stats = db.get_sentiment_stats()
            if not stats.empty:
                total_reviews = stats['count'].sum()
                st.metric("Total Reviews", total_reviews)
                
                # Sentiment distribution
                sentiment_data = {row['sentiment_label']: row['count'] for _, row in stats.iterrows()}
                for sentiment, count in sentiment_data.items():
                    st.metric(
                        f"{sentiment.title()}",
                        count,
                        f"{(count/total_reviews*100):.0f}%"
                    )
            else:
                st.info("No data yet")
        except Exception as e:
            st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("### 🎯 Analysis Mode")
        
        analysis_mode = st.selectbox(
            "Select Analysis Depth:",
            ["Standard", "Advanced", "Comprehensive"],
            index=1
        )
        st.session_state.analysis_mode = analysis_mode
    
    # Main dashboard dengan lebih banyak tabs
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
    """Enhanced dashboard dengan lebih banyak insights"""
    st.subheader("📊 Advanced Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Real-time review stream
        st.markdown("#### 📝 Recent Review Analysis")
        
        try:
            reviews = db.get_reviews(8)
            if not reviews.empty:
                for i, row in reviews.iterrows():
                    with st.container():
                        sentiment = row.get('sentiment_label', 'Not analyzed')
                        confidence = row.get('confidence_score', 0)
                        
                        # Color-coded sentiment
                        if sentiment == "positive":
                            sentiment_color = "🟢"
                        elif sentiment == "negative":
                            sentiment_color = "🔴"
                        else:
                            sentiment_color = "🟡"
                        
                        col_a, col_b, col_c = st.columns([1, 3, 1])
                        
                        with col_a:
                            st.write(f"{sentiment_color} **{sentiment.title()}**")
                            st.write(f"_{confidence:.0%} conf._")
                        
                        with col_b:
                            st.write(f"**{row['product_id']}** - {row['rating']}⭐")
                            review_preview = row['review_text']
                            if len(review_preview) > 80:
                                review_preview = review_preview[:80] + "..."
                            st.write(review_preview)
                        
                        with col_c:
                            st.write(f"📅 {row.get('created_date', 'N/A')}")
                        
                        st.markdown("---")
            else:
                st.info("No reviews available")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with col2:
        # Advanced metrics
        st.markdown("#### 📈 Performance Metrics")
        
        try:
            stats = db.get_sentiment_stats()
            if not stats.empty:
                # Sentiment distribution chart
                fig = px.pie(
                    stats, 
                    values='count', 
                    names='sentiment_label',
                    color='sentiment_label',
                    color_discrete_map={
                        'positive': '#00cc96',
                        'negative': '#ef553b', 
                        'neutral': '#ffa15a'
                    },
                    hole=0.5
                )
                fig.update_layout(
                    title="Sentiment Distribution",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence metrics
                avg_confidence = stats['avg_confidence'].mean()
                total_reviews = stats['count'].sum()
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Avg Confidence", f"{avg_confidence:.0%}")
                with col_met2:
                    st.metric("Total Analyzed", total_reviews)
            else:
                st.info("No data for metrics")
        except Exception as e:
            st.error(f"Error: {e}")

def display_trends():
    """Enhanced trends analysis"""
    st.subheader("📈 Advanced Trend Analysis")
    
    # Time period selector
    col_period, col_granularity = st.columns(2)
    
    with col_period:
        days = st.slider("Analysis Period (days):", 7, 180, 30)
    
    with col_granularity:
        granularity = st.selectbox("Granularity:", ["Daily", "Weekly", "Monthly"])
    
    try:
        trends = db.get_sentiment_trends(days)
        
        if not trends.empty:
            # Convert to appropriate granularity
            if granularity == "Weekly":
                trends['week'] = pd.to_datetime(trends['date']).dt.to_period('W').dt.start_time
                trends = trends.groupby('week').agg({
                    'positive_count': 'sum',
                    'negative_count': 'sum',
                    'neutral_count': 'sum',
                    'total_reviews': 'sum'
                }).reset_index()
                trends['positive_percentage'] = (trends['positive_count'] / trends['total_reviews'] * 100).round(2)
                trends['negative_percentage'] = (trends['negative_count'] / trends['total_reviews'] * 100).round(2)
                x_axis = 'week'
            elif granularity == "Monthly":
                trends['month'] = pd.to_datetime(trends['date']).dt.to_period('M').dt.start_time
                trends = trends.groupby('month').agg({
                    'positive_count': 'sum',
                    'negative_count': 'sum', 
                    'neutral_count': 'sum',
                    'total_reviews': 'sum'
                }).reset_index()
                trends['positive_percentage'] = (trends['positive_count'] / trends['total_reviews'] * 100).round(2)
                trends['negative_percentage'] = (trends['negative_count'] / trends['total_reviews'] * 100).round(2)
                x_axis = 'month'
            else:
                x_axis = 'date'
            
            # Create advanced trend visualization
            fig = go.Figure()
            
            # Positive trend
            fig.add_trace(go.Scatter(
                x=trends[x_axis], 
                y=trends['positive_percentage'],
                mode='lines+markers',
                name='Positive %',
                line=dict(color='#00cc96', width=4),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)'
            ))
            
            # Negative trend
            fig.add_trace(go.Scatter(
                x=trends[x_axis], 
                y=trends['negative_percentage'],
                mode='lines+markers', 
                name='Negative %',
                line=dict(color='#ef553b', width=4),
                fill='tozeroy',
                fillcolor='rgba(239, 85, 59, 0.1)'
            ))
            
            fig.update_layout(
                title=f"Sentiment Trends ({granularity})",
                xaxis_title="Time",
                yaxis_title="Percentage (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                positive_trend = trends['positive_percentage'].iloc[-1] - trends['positive_percentage'].iloc[0]
                st.metric(
                    "Positive Trend", 
                    f"{trends['positive_percentage'].iloc[-1]:.1f}%",
                    f"{positive_trend:+.1f}%"
                )
            
            with col2:
                negative_trend = trends['negative_percentage'].iloc[-1] - trends['negative_percentage'].iloc[0]
                st.metric(
                    "Negative Trend",
                    f"{trends['negative_percentage'].iloc[-1]:.1f}%", 
                    f"{negative_trend:+.1f}%"
                )
            
            with col3:
                total_reviews = trends['total_reviews'].sum()
                st.metric("Total Reviews", f"{total_reviews:,}")
            
            with col4:
                avg_daily = total_reviews / max(1, len(trends))
                st.metric("Avg Daily", f"{avg_daily:.1f}")
            
        else:
            st.info("No trend data available. Analyze some reviews first.")
    except Exception as e:
        st.error(f"Error loading trends: {e}")

def display_products():
    """Enhanced product analysis"""
    st.subheader("🏆 Product Performance Analytics")
    
    try:
        product_stats = db.get_product_stats()
        
        if not product_stats.empty:
            # Top products ranking
            st.markdown("#### 📊 Product Ranking")
            
            # Enhanced product table
            display_df = product_stats.copy()
            display_df['sentiment_ratio'] = (display_df['positive_count'] / display_df['total_reviews'] * 100).round(1)
            display_df = display_df.sort_values('sentiment_ratio', ascending=False)
            
            st.dataframe(
                display_df.head(10),
                use_container_width=True,
                column_config={
                    "product_id": "Product ID",
                    "total_reviews": "Reviews",
                    "avg_rating": "Avg Rating",
                    "positive_count": "👍 Positive",
                    "negative_count": "👎 Negative", 
                    "neutral_count": "😐 Neutral",
                    "sentiment_ratio": "😊 Positive %"
                }
            )
            
            # Product comparison chart
            st.markdown("#### 📈 Product Comparison")
            
            top_products = product_stats.nlargest(5, 'total_reviews')
            
            fig = go.Figure()
            
            for _, product in top_products.iterrows():
                fig.add_trace(go.Bar(
                    name=product['product_id'][:10] + "...",
                    x=['Positive', 'Neutral', 'Negative'],
                    y=[product['positive_count'], product['neutral_count'], product['negative_count']],
                    text=[product['positive_count'], product['neutral_count'], product['negative_count']],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title="Top Products: Sentiment Distribution",
                barmode='group',
                height=400,
                xaxis_title="Sentiment",
                yaxis_title="Number of Reviews"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No product data available")
    except Exception as e:
        st.error(f"Error loading product stats: {e}")

def display_ai_insights():
    """AI model insights and performance"""