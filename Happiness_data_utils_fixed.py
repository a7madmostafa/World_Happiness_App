import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_data(file_path_or_buffer) -> pd.DataFrame:
    """Load and preprocess the World Happiness Report dataset."""
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None

    if isinstance(file_path_or_buffer, str):
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path_or_buffer, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                raise FileNotFoundError(f"❌ Could not find file: {file_path_or_buffer}")
    else:
        df = pd.read_csv(file_path_or_buffer)

    if df is None or df.empty:
        raise ValueError(f"❌ Could not read CSV or file is empty")

    column_mapping = {}
    used_mappings = set()
    mapping_rules = [
        (['country name', 'country'], 'Country'),
        (['regional indicator', 'region'], 'Region'),
        (['ladder score'], 'Happiness_Score'),
        (['logged gdp per capita'], 'GDP_per_capita'),
        (['social support'], 'Social_support'),
        (['healthy life expectancy'], 'Healthy_life_expectancy'),
        (['freedom to make life choices', 'freedom'], 'Freedom'),
        (['generosity'], 'Generosity'),
        (['perceptions of corruption'], 'Perceptions_of_corruption'),
        (['dystopia + residual', 'dystopia residual'], 'Dystopia_residual'),
    ]
    for col in df.columns:
        col_lower = col.lower().strip()
        for keywords, target_name in mapping_rules:
            if any(keyword in col_lower for keyword in keywords) and target_name not in used_mappings:
                column_mapping[col] = target_name
                used_mappings.add(target_name)
                break
    if column_mapping:
        df = df.rename(columns=column_mapping)

    required_cols = ['Country', 'Happiness_Score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Required column missing: {col}")

    df = df.dropna(subset=['Country', 'Happiness_Score']).drop_duplicates(subset=['Country'])

    numeric_columns = [
        'Happiness_Score', 'GDP_per_capita', 'Social_support',
        'Healthy_life_expectancy', 'Freedom', 'Generosity',
        'Perceptions_of_corruption', 'Dystopia_residual'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Happiness_Rank' not in df.columns:
        df['Happiness_Rank'] = df['Happiness_Score'].rank(method='dense', ascending=False).astype(int)

    if 'Region' not in df.columns:
        df['Region'] = 'Unknown Region'
    else:
        df['Region'] = df['Region'].fillna('Unknown Region')

    df['Country'] = df['Country'].str.strip()
    df = df.sort_values('Happiness_Score', ascending=False).reset_index(drop=True)

    return df

def get_country_list(df: pd.DataFrame) -> List[str]:
    return sorted(df['Country'].dropna().unique().tolist()) if not df.empty else []

def get_region_list(df: pd.DataFrame) -> List[str]:
    return sorted(df['Region'].dropna().unique().tolist()) if 'Region' in df.columns else []

def filter_by_regions(df: pd.DataFrame, regions: List[str]) -> pd.DataFrame:
    return df[df['Region'].isin(regions)] if regions else df

def get_top_countries(df: pd.DataFrame, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    if ascending:
        return df.nsmallest(n, 'Happiness_Score').reset_index(drop=True)
    return df.nlargest(n, 'Happiness_Score').reset_index(drop=True)

def get_regional_stats(df: pd.DataFrame) -> pd.DataFrame:
    factor_columns = [
        'Happiness_Score', 'GDP_per_capita', 'Social_support',
        'Healthy_life_expectancy', 'Freedom', 'Generosity',
        'Perceptions_of_corruption'
    ]
    available_cols = [c for c in factor_columns if c in df.columns]
    return df.groupby('Region')[available_cols].agg(['mean', 'std', 'count']).round(3)

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    factor_columns = [
        'Happiness_Score', 'GDP_per_capita', 'Social_support',
        'Healthy_life_expectancy', 'Freedom', 'Generosity',
        'Perceptions_of_corruption'
    ]
    available_cols = [c for c in factor_columns if c in df.columns]
    return df[available_cols].corr().round(3)

def get_summary_stats(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}
    return {
        'total_countries': len(df),
        'total_regions': len(df['Region'].unique()),
        'avg_happiness': round(df['Happiness_Score'].mean(), 2),
        'happiness_range': (round(df['Happiness_Score'].min(), 2), round(df['Happiness_Score'].max(), 2)),
        'happiest_country': df.loc[df['Happiness_Score'].idxmax(), 'Country'],
        'least_happy_country': df.loc[df['Happiness_Score'].idxmin(), 'Country'],
        'most_common_region': df['Region'].mode().iloc[0]
    }

def create_world_map(df: pd.DataFrame) -> go.Figure:
    return px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color='Happiness_Score',
        hover_name='Country',
        hover_data=['Region', 'Happiness_Rank'],
        color_continuous_scale='RdYlGn',
        title='World Happiness Score Map'
    )

def create_ranking_chart(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = df.head(top_n)
    return px.bar(
        data,
        y='Country',
        x='Happiness_Score',
        orientation='h',
        color='Happiness_Score',
        color_continuous_scale='RdYlGn',
        title=f'Country Rankings (Top {top_n})'
    )

def create_factor_analysis_chart(df: pd.DataFrame, countries: List[str]) -> go.Figure:
    factors = ['GDP_per_capita', 'Social_support', 'Healthy_life_expectancy',
               'Freedom', 'Generosity', 'Perceptions_of_corruption']
    fig = go.Figure()
    max_values = df[factors].max()
    for country in countries:
        row = df[df['Country'] == country].iloc[0]
        values = [(row[col] / max_values[col]) if max_values[col] > 0 else 0 for col in factors]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[col.replace('_', ' ').title() for col in factors],
            fill='toself',
            name=country
        ))
    fig.update_layout(title='Happiness Factors Comparison (Normalized)')
    return fig

def create_regional_comparison(df: pd.DataFrame) -> go.Figure:
    regional_stats = df.groupby('Region')['Happiness_Score'].mean().reset_index()
    return px.bar(
        regional_stats,
        x='Region',
        y='Happiness_Score',
        color='Happiness_Score',
        color_continuous_scale='RdYlGn',
        title='Average Happiness Score by Region'
    )

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr_matrix = calculate_correlations(df)
    return px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Correlation Matrix: Happiness Factors"
    )

def create_scatter_plot(df: pd.DataFrame, x_factor: str, y_factor: str = 'Happiness_Score') -> go.Figure:
    return px.scatter(
        df,
        x=x_factor,
        y=y_factor,
        color='Region',
        size='Happiness_Score',
        hover_name='Country',
        trendline="ols",
        title=f'Relationship between {x_factor} and {y_factor}'
    )

def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Creates a histogram to visualize the distribution of a column."""
    try:
        if df is None or df.empty:
            return go.Figure().add_annotation(text="No data available for distribution plot",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        if column not in df.columns:
            return go.Figure().add_annotation(text=f"Column '{column}' not available",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        clean_data = df[column].dropna()
        if clean_data.empty:
            return go.Figure().add_annotation(text=f"No valid data for column '{column}'",
                                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        fig = px.histogram(clean_data, x=column,
                           nbins=min(30, max(10, len(clean_data) // 5)),
                           title=f'Distribution of {column.replace("_", " ").title()}',
                           labels={column: column.replace("_", " ").title()},
                           color_discrete_sequence=['skyblue'])

        mean_val = clean_data.mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_val:.2f}")

        fig.update_layout(showlegend=False, title_x=0.5)
        return fig

    except Exception as e:
        logger.error(f"Error creating distribution plot: {e}")
        return go.Figure().add_annotation(text=f"Error creating distribution: {str(e)}",
                                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
