import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
try:
    from Happiness_data_utils_fixed import (
        load_data,
        get_country_list,
        get_region_list,
        get_top_countries,
        get_regional_stats,
        calculate_correlations,
        create_world_map,
        create_ranking_chart,
        create_factor_analysis_chart,
        create_regional_comparison,
        create_correlation_heatmap,
        create_scatter_plot,
        create_distribution_plot,
        get_summary_stats,
        filter_by_regions
    )
except ImportError as e:
    st.error(f"❌ Error importing Happiness_data_utils: {e}")
    st.error("Please ensure Happiness_data_utils.py is in the same directory as this file.")
    st.stop()

st.set_page_config(
    page_title="World Happiness Report 2021",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        padding-top: 2rem;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #6b7280;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        line-height: 1;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    .stSelectbox > div > div > div {
        border-radius: 8px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }

    .plot-container {
        border-radius: 12px;
        padding: 1rem;
        background: white;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

def display_error_message(message: str, error_type: str = "Error"):
    """Display a formatted error message."""
    st.error(f"❌ **{error_type}**: {message}")

def display_success_message(message: str):
    """Display a formatted success message."""
    st.success(f"✅ {message}")

def display_info_message(message: str):
    """Display a formatted info message."""
    st.info(f"ℹ️ {message}")

def safe_execute(func, *args, error_msg="An error occurred", **kwargs):
    """Safely execute a function and handle errors gracefully."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {str(e)}")
        display_error_message(f"{error_msg}: {str(e)}")
        return None

@st.cache_data
def load_data_safe(file_path_or_buffer):
    """Safely load data with error handling."""
    return safe_execute(load_data, file_path_or_buffer, 
                       error_msg="Failed to load data")

def attempt_data_load():
    """Attempt to load data from various common file paths."""
    possible_paths = [
        "world-happiness-report-2021.csv",
        "data/world-happiness-report-2021.csv",
        "datasets/world-happiness-report-2021.csv",
        "happiness-2021.csv",
        "happiness_report_2021.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = load_data_safe(path)
            if df is not None:
                display_success_message(f"Data loaded successfully from: {path}")
                return df
    
    return None


if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

df = None
if not st.session_state.data_loaded:
    with st.spinner("🔄 Loading data..."):
        df = attempt_data_load()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
else:
    df = st.session_state.df

if df is None or df.empty:
    st.markdown("<h1 class='main-header'>World Happiness Report 2021</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Data Dashboard</p>", unsafe_allow_html=True)
    
    st.warning("📁 **Data file not found!** Please upload your dataset to continue.")
    
    with st.expander("📋 Data Requirements", expanded=True):
        st.markdown("""
        **Required columns (will be auto-mapped):**
        - Country names
        - Happiness scores/ladder scores
        - Optional: Region, GDP, Social support, Life expectancy, Freedom, Generosity, Corruption perception
        
        **Supported formats:** CSV files with UTF-8, Latin-1, or CP1252 encoding
        
        **Sample data sources:**
        - [World Happiness Report 2021 (Kaggle)](https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021)
        - [UN World Happiness Report](https://worldhappiness.report/)
        """)
    
    uploaded_file = st.file_uploader(
        "📤 Upload your World Happiness Report CSV file",
        type=['csv'],
        help="Select a CSV file containing happiness data"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing uploaded file..."):
            try:
                # Reset session state before loading new file
                st.session_state.data_loaded = False
                st.session_state.df = None
                
                df = load_data_safe(uploaded_file)
                if df is not None and not df.empty:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    display_success_message("File uploaded and processed successfully!")
                    st.rerun()
                else:
                    display_error_message("Failed to process the uploaded file. Please check the file format and try again.")
                    st.error("Please ensure your CSV file contains the required columns (Country and Happiness Score)")
            except Exception as e:
                display_error_message(f"Error processing uploaded file: {str(e)}")
                st.error("Debug info: Check that your CSV file is properly formatted and contains valid data")
    else:
        st.info("👆 Please upload a CSV file to begin exploring happiness data.")
        st.stop()

def create_sidebar():
    """Create and return sidebar navigation."""
    with st.sidebar:
        st.markdown("## 🌍 Navigation")
        
        pages = {
            "🏠 Overview": "home",
            "🏆 Country Rankings": "rankings",
            "📊 Factors Analysis": "factors",
            "🌏 Regional Insights": "regions",
            "📈 Trends & Correlations": "trends",
            "🔍 Data Explorer": "explorer",
        }
        
        selected_page = st.radio("Choose a page:", list(pages.keys()), index=0)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Summary")
        
        if df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Countries", len(df))
                st.metric("Regions", len(df['Region'].unique()))
            with col2:
                st.metric("Avg Score", f"{df['Happiness_Score'].mean():.2f}")
                st.metric("Score Range", 
                         f"{df['Happiness_Score'].min():.1f} - {df['Happiness_Score'].max():.1f}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard analyzes the **World Happiness Report 2021** data, 
        exploring factors that contribute to national happiness and well-being.
        """)
        
        return pages[selected_page]

current_page = create_sidebar()

if current_page == "home":
    st.markdown("<h1 class='main-header'>World Happiness Report 2021</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Exploring Global Well-being and Its Key Drivers</p>", unsafe_allow_html=True)

    stats = safe_execute(get_summary_stats, df, error_msg="Failed to calculate summary statistics")
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("Average Happiness", f"{stats['avg_happiness']:.2f}", col1),
            ("Happiest Country", stats['happiest_country'], col2),
            ("Least Happy Country", stats['least_happy_country'], col3),
            ("Total Countries", str(stats['total_countries']), col4)
        ]
        
        for label, value, col in metrics:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    
    
    st.subheader("🗺️ Interactive World Happiness Map")
    st.write("Explore happiness scores across different countries. Hover over countries to see detailed information.")
    
    with st.container():
        fig_map = safe_execute(create_world_map, df, error_msg="Failed to create world map")
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)

    
    if stats:
        st.markdown("---")
        st.subheader("🔍 Quick Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **📈 Happiness Range:** {stats['happiness_range'][0]} - {stats['happiness_range'][1]}
            
            **🏆 Top Performer:** {stats['happiest_country']}
            
            **📍 Most Common Region:** {stats['most_common_region']}
            """)
        
        with col2:
            # Show top 5 and bottom 5 countries
            if len(df) >= 10:
                top_5 = df.head(5)['Country'].tolist()
                bottom_5 = df.tail(5)['Country'].tolist()
                
                st.markdown("**🥇 Top 5 Happiest:**")
                for i, country in enumerate(top_5, 1):
                    st.write(f"{i}. {country}")
                
                st.markdown("**🥺 Bottom 5:**")
                for i, country in enumerate(bottom_5, 1):
                    st.write(f"{i}. {country}")


elif current_page == "rankings":
    st.markdown("<h1 class='main-header'>🏆 Country Rankings</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Official ranking of countries by their happiness score</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🎛️ Controls")
        top_n = st.slider("Number of countries to display", 5, min(100, len(df)), 20)
        sort_order = st.radio("Sorting order", ["Happiest First", "Least Happy First"])
        
        # Display options
        show_region = st.checkbox("Show regions", value=True)
        show_rank = st.checkbox("Show ranks", value=True)
    
    ascending = sort_order == "Least Happy First"
    top_countries_df = safe_execute(get_top_countries, df, top_n, ascending, 
                                   error_msg="Failed to get top countries")
    
    if top_countries_df is not None:
        with col2:
            st.markdown("### 📊 Rankings Chart")
            fig_ranking = safe_execute(create_ranking_chart, top_countries_df, top_n,
                                     error_msg="Failed to create ranking chart")
            if fig_ranking:
                st.plotly_chart(fig_ranking, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Full Rankings Table")
        
        # Prepare columns for display
        display_cols = ['Country', 'Happiness_Score']
        if show_rank and 'Happiness_Rank' in df.columns:
            display_cols.insert(0, 'Happiness_Rank')
        if show_region and 'Region' in df.columns:
            display_cols.append('Region')
        
        # Display the table
        st.dataframe(
            df[display_cols].style.format({
                'Happiness_Score': '{:.3f}',
            }).background_gradient(subset=['Happiness_Score'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )

elif current_page == "factors":
    st.markdown("<h1 class='main-header'>📊 Factors Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Compare key happiness factors across selected countries</p>", unsafe_allow_html=True)
    
    available_countries = safe_execute(get_country_list, df, error_msg="Failed to get country list")
    
    if available_countries:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Country Selection")
            
            
            preset_options = {
                "Top 5 Happiest": df.head(5)['Country'].tolist() if len(df) >= 5 else [],
                "Bottom 5 Happiest": df.tail(5)['Country'].tolist() if len(df) >= 5 else [],
                "Nordic Countries": [c for c in available_countries if c in ['Finland', 'Denmark', 'Norway', 'Sweden', 'Iceland']],
                "Custom Selection": []
            }
            
            preset = st.selectbox("Choose preset or custom", list(preset_options.keys()))
            
            if preset == "Custom Selection":
                selected_countries = st.multiselect(
                    "Select countries for comparison",
                    available_countries,
                    default=available_countries[:3] if len(available_countries) >= 3 else available_countries,
                    help="Select 2-8 countries for optimal visualization"
                )
            else:
                selected_countries = preset_options[preset]
                st.write(f"**Selected:** {', '.join(selected_countries)}")
            

            if len(selected_countries) > 8:
                st.warning("⚠️ Too many countries selected. Showing first 8 for better visualization.")
                selected_countries = selected_countries[:8]
        
        with col2:
            if selected_countries:
                st.markdown("### 🕸️ Factors Comparison")
                fig_radar = safe_execute(create_factor_analysis_chart, df, selected_countries,
                                       error_msg="Failed to create factor analysis chart")
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
            
                st.markdown("### 📋 Detailed Comparison")
                factor_columns = ['Country', 'Happiness_Score', 'GDP_per_capita', 'Social_support', 
                                'Healthy_life_expectancy', 'Freedom', 'Generosity', 'Perceptions_of_corruption']
                available_factor_cols = [col for col in factor_columns if col in df.columns]
                
                comparison_df = df[df['Country'].isin(selected_countries)][available_factor_cols]
                st.dataframe(
                    comparison_df.style.format({col: '{:.3f}' for col in available_factor_cols[1:] if col in df.columns}),
                    use_container_width=True
                )
            else:
                st.info("👆 Please select countries to view the analysis.")

elif current_page == "regions":
    st.markdown("<h1 class='main-header'>🌏 Regional Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Compare happiness levels and factors across different world regions</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Regional Happiness Comparison")
        fig_regions = safe_execute(create_regional_comparison, df,
                                 error_msg="Failed to create regional comparison")
        if fig_regions:
            st.plotly_chart(fig_regions, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Regional Statistics")
        regional_stats_df = safe_execute(get_regional_stats, df,
                                       error_msg="Failed to calculate regional statistics")
        if regional_stats_df is not None and not regional_stats_df.empty:
            mean_stats = regional_stats_df.xs('mean', axis=1, level=1)
            st.dataframe(
                mean_stats.style.format('{:.3f}').background_gradient(cmap='RdYlGn'),
                use_container_width=True
            )
        else:
            st.error("No regional statistics available")

    st.markdown("---")
    st.subheader("🗂️ Countries by Region")
    
    region_counts = df.groupby('Region').size().sort_values(ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_region_dist = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title="Distribution of Countries by Region"
        )
        fig_region_dist.update_layout(title_x=0.5)
        st.plotly_chart(fig_region_dist, use_container_width=True)
    
    with col2:
        st.markdown("**Country Count by Region:**")
        for region, count in region_counts.items():
            st.write(f"• **{region}**: {count} countries")

elif current_page == "trends":
    st.markdown("<h1 class='main-header'>📈 Trends & Correlations</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore relationships between different happiness factors</p>", unsafe_allow_html=True)
    
    # Correlation Matrix
    st.subheader("🔥 Correlation Heatmap")
    st.write("This heatmap shows how different factors correlate with each other and with happiness scores.")
    
    fig_corr = safe_execute(create_correlation_heatmap, df,
                           error_msg="Failed to create correlation heatmap")
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter Plot Analysis
    st.subheader("🎯 Factor Analysis")
    st.write("Analyze how individual factors correlate with happiness scores.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        factors = [
            'GDP_per_capita', 'Social_support', 'Healthy_life_expectancy',
            'Freedom', 'Generosity', 'Perceptions_of_corruption'
        ]
        available_factors = [f for f in factors if f in df.columns]
        
        if available_factors:
            x_factor = st.selectbox(
                "Select factor to analyze",
                available_factors,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            show_regions = st.checkbox("Color by region", value=True)
            show_trendline = st.checkbox("Show trend line", value=True)
        else:
            st.error("No factors available for analysis")
            x_factor = None
    
    with col2:
        if x_factor:
            fig_scatter = safe_execute(create_scatter_plot, df, x_factor,
                                     error_msg="Failed to create scatter plot")
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Show correlation coefficient
            if 'Happiness_Score' in df.columns:
                corr_coef = df[x_factor].corr(df['Happiness_Score'])
                if not pd.isna(corr_coef):
                    st.metric(
                        "Correlation with Happiness",
                        f"{corr_coef:.3f}",
                        delta=None,
                        help="Values closer to 1 or -1 indicate stronger correlation"
                    )


elif current_page == "explorer":
    st.markdown("<h1 class='main-header'>🔍 Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Filter and explore the dataset in detail</p>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("🎛️ Filters & Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_regions = safe_execute(get_region_list, df, error_msg="Failed to get region list")
        if available_regions:
            selected_regions = st.multiselect(
                "Filter by Region(s)",
                available_regions,
                default=[],
                help="Leave empty to show all regions"
            )
        else:
            selected_regions = []
    
    with col2:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_column = st.selectbox(
            "Column for distribution",
            numeric_columns,
            help="Select a numeric column to visualize its distribution"
        )
    
    with col3:
        if len(df) > 0:
            score_range = st.slider(
                "Happiness Score Range",
                float(df['Happiness_Score'].min()),
                float(df['Happiness_Score'].max()),
                (float(df['Happiness_Score'].min()), float(df['Happiness_Score'].max())),
                step=0.1
            )
        else:
            score_range = (0.0, 10.0)
    
    
    filtered_df = df.copy()
    
    if selected_regions:
        filtered_df = safe_execute(filter_by_regions, filtered_df, selected_regions,
                                 error_msg="Failed to filter by regions") or filtered_df
    
    
    filtered_df = filtered_df[
        (filtered_df['Happiness_Score'] >= score_range[0]) & 
        (filtered_df['Happiness_Score'] <= score_range[1])
    ]
    
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not filtered_df.empty and selected_column:
            st.subheader(f"📊 Distribution of {selected_column.replace('_', ' ').title()}")
            fig_dist = safe_execute(create_distribution_plot, filtered_df, selected_column,
                                  error_msg="Failed to create distribution plot")
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            if filtered_df.empty:
                st.warning("⚠️ No data found for the selected filters.")
            else:
                st.info("ℹ️ Please select a column to see the distribution.")
    
    with col2:
        st.subheader("📈 Filter Summary")
        st.metric("Countries Shown", len(filtered_df))
        st.metric("Original Total", len(df))
        
        if len(filtered_df) > 0:
            st.metric("Avg Happiness", f"{filtered_df['Happiness_Score'].mean():.2f}")
            st.metric("Score Range", 
                     f"{filtered_df['Happiness_Score'].min():.1f} - {filtered_df['Happiness_Score'].max():.1f}")
    
    
    st.markdown("---")
    st.subheader("📋 Filtered Data Table")
    
    if not filtered_df.empty:
        # Column selector for the table
        all_columns = filtered_df.columns.tolist()
        default_columns = ['Country', 'Happiness_Score', 'Happiness_Rank', 'Region']
        default_columns = [col for col in default_columns if col in all_columns]
        
        selected_table_columns = st.multiselect(
            "Select columns to display",
            all_columns,
            default=default_columns,
            help="Choose which columns to show in the table below"
        )
        
        if selected_table_columns:
            st.dataframe(
                filtered_df[selected_table_columns].style.format({
                    col: '{:.3f}' for col in selected_table_columns 
                    if col in filtered_df.select_dtypes(include=[np.number]).columns
                }),
                use_container_width=True,
                height=400
            )
            
    
            csv = filtered_df[selected_table_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download filtered data as CSV",
                data=csv,
                file_name="filtered_happiness_data.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Please select at least one column to display the table.")
    else:
        st.warning("No data to display with current filters.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p>🌍 <strong>World Happiness Report 2021 Dashboard</strong></p>
    <p>Built with ❤️ using Streamlit and Plotly</p>
    <p style='font-size: 0.8rem;'>Data source: World Happiness Report | UN Sustainable Development Solutions Network</p>
</div>
""", unsafe_allow_html=True)