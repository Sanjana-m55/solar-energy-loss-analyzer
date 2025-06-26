"""
Visualization Module for Solar Energy Loss Analysis

This module provides comprehensive visualization capabilities for solar PV
plant performance analysis, including interactive plots, dashboards, and
advanced visualizations for loss attribution.

Author: AI Assistant
Date: June 25, 2025
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class Visualizer:
    """
    Advanced visualization class for solar energy analysis
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.color_palette = {
            'cloud': '#FF6B6B',
            'shading': '#4ECDC4', 
            'temperature': '#45B7D1',
            'soiling': '#FFA07A',
            'other': '#98D8C8',
            'actual': '#2E8B57',
            'theoretical': '#FF8C00'
        }
        
    def create_loss_breakdown_chart(self, loss_data, title="Energy Loss Breakdown"):
        """
        Create a stacked bar chart for loss breakdown
        
        Args:
            loss_data (pd.DataFrame): Loss breakdown data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Stacked bar chart
        """
        fig = go.Figure()
        
        loss_categories = ['Cloud Cover', 'Shading', 'Temperature', 'Soiling', 'Other Losses']
        colors = [self.color_palette['cloud'], self.color_palette['shading'], 
                 self.color_palette['temperature'], self.color_palette['soiling'], 
                 self.color_palette['other']]
        
        for i, category in enumerate(loss_categories):
            if category in loss_data.columns:
                fig.add_trace(go.Bar(
                    x=loss_data.index,
                    y=loss_data[category],
                    name=category,
                    marker_color=colors[i],
                    hovertemplate=f'<b>{category}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Loss: %{y:.2f} kWh<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Energy Loss (kWh)",
            barmode='stack',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_loss_pie_chart(self, loss_summary, title="Loss Distribution"):
        """
        Create a pie chart for loss distribution
        
        Args:
            loss_summary (pd.DataFrame): Loss summary data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Pie chart
        """
        fig = go.Figure(data=[go.Pie(
            labels=loss_summary['Loss Type'],
            values=loss_summary['Total Loss (kWh)'],
            hole=.3,
            marker_colors=[self.color_palette['cloud'], self.color_palette['shading'],
                          self.color_palette['temperature'], self.color_palette['soiling'],
                          self.color_palette['other']],
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>' +
                         'Loss: %{value:.2f} kWh<br>' +
                         'Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_time_series_plot(self, data, columns, title="Time Series Analysis"):
        """
        Create time series plot for multiple variables
        
        Args:
            data (pd.DataFrame): Time series data
            columns (list): Columns to plot
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Time series plot
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(columns):
            if col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index if isinstance(data.index, pd.DatetimeIndex) else range(len(data)),
                    y=data[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{col}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Value: %{y:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_heatmap(self, data, title="Performance Heatmap"):
        """
        Create a heatmap for performance analysis
        
        Args:
            data (pd.DataFrame): Data for heatmap
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        if isinstance(data.index, pd.DatetimeIndex):
            # Create hour vs day heatmap
            data_pivot = data.copy()
            data_pivot['hour'] = data_pivot.index.hour
            data_pivot['day'] = data_pivot.index.day
            
            # Pivot for heatmap
            heatmap_data = data_pivot.pivot_table(
                values=data.columns[0] if len(data.columns) > 0 else 'value',
                index='hour',
                columns='day',
                aggfunc='mean'
            )
        else:
            heatmap_data = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlBu_r',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            xaxis_title="Day" if isinstance(data.index, pd.DatetimeIndex) else "Variable",
            yaxis_title="Hour" if isinstance(data.index, pd.DatetimeIndex) else "Variable"
        )
        
        return fig
    
    def create_scatter_plot(self, data, x_col, y_col, color_col=None, size_col=None, title="Scatter Plot"):
        """
        Create an interactive scatter plot
        
        Args:
            data (pd.DataFrame): Data for scatter plot
            x_col (str): X-axis column
            y_col (str): Y-axis column
            color_col (str): Color column (optional)
            size_col (str): Size column (optional)
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot
        """
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            hover_data=data.columns.tolist(),
            height=500
        )
        
        fig.update_traces(
            marker=dict(
                sizemode='diameter',
                sizeref=2.*max(data[size_col] if size_col else [1])/(40.**2),
                sizemin=4
            )
        )
        
        return fig
    
    def create_box_plot(self, data, x_col, y_col, title="Box Plot Analysis"):
        """
        Create a box plot for distribution analysis
        
        Args:
            data (pd.DataFrame): Data for box plot
            x_col (str): X-axis column (categorical)
            y_col (str): Y-axis column (numerical)
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Box plot
        """
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            title=title,
            height=500
        )
        
        fig.update_traces(
            marker=dict(
                outliercolor='rgba(219, 64, 82, 0.6)',
                line=dict(
                    outliercolor='rgba(219, 64, 82, 0.6)',
                    outlierwidth=2
                )
            )
        )
        
        return fig
    
    def create_performance_dashboard(self, loss_data, asset_data, time_data):
        """
        Create a comprehensive performance dashboard
        
        Args:
            loss_data (pd.DataFrame): Loss breakdown data
            asset_data (pd.DataFrame): Asset performance data
            time_data (pd.DataFrame): Time series data
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Loss Breakdown Over Time",
                "Asset Performance Comparison", 
                "Daily Loss Patterns",
                "Efficiency Trends"
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Plot 1: Loss breakdown
        if not loss_data.empty:
            for i, col in enumerate(loss_data.columns):
                fig.add_trace(
                    go.Scatter(
                        x=loss_data.index,
                        y=loss_data[col],
                        name=col,
                        stackgroup='one',
                        mode='lines'
                    ),
                    row=1, col=1
                )
        
        # Plot 2: Asset performance
        if not asset_data.empty and 'Asset' in asset_data.columns and 'Efficiency' in asset_data.columns:
            fig.add_trace(
                go.Bar(
                    x=asset_data['Asset'],
                    y=asset_data['Efficiency'],
                    name='Efficiency',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Daily patterns (if time data available)
        if not time_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=time_data.index,
                    y=time_data.iloc[:, 0] if len(time_data.columns) > 0 else [0],
                    name='Daily Pattern',
                    mode='lines+markers',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Efficiency trends
        if not time_data.empty and len(time_data.columns) > 1:
            fig.add_trace(
                go.Scatter(
                    x=time_data.index,
                    y=time_data.iloc[:, 1],
                    name='Efficiency Trend',
                    mode='lines',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Solar Plant Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_asset_ranking_chart(self, asset_data, metric='Efficiency', title="Asset Performance Ranking"):
        """
        Create a ranking chart for asset performance
        
        Args:
            asset_data (pd.DataFrame): Asset performance data
            metric (str): Metric to rank by
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Ranking chart
        """
        # Sort by metric
        sorted_data = asset_data.sort_values(metric, ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=sorted_data[metric],
            y=sorted_data.index if 'Asset' not in sorted_data.columns else sorted_data['Asset'],
            orientation='h',
            marker=dict(
                color=sorted_data[metric],
                colorscale='RdYlGn',
                colorbar=dict(title=metric)
            ),
            text=sorted_data[metric].round(2),
            textposition='outside',
            hovertemplate=f'<b>%{{y}}</b><br>{metric}: %{{x:.2f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=metric,
            yaxis_title="Asset",
            height=max(400, len(sorted_data) * 30),
            showlegend=False
        )
        
        return fig
    
    def create_correlation_matrix(self, data, title="Correlation Matrix"):
        """
        Create a correlation matrix heatmap
        
        Args:
            data (pd.DataFrame): Data for correlation analysis
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        # Calculate correlation matrix
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_data, title="Feature Importance"):
        """
        Create a feature importance chart
        
        Args:
            importance_data (pd.DataFrame): Feature importance data
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Feature importance chart
        """
        # Sort by importance
        sorted_data = importance_data.sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=sorted_data['Importance'],
            y=sorted_data['Feature'],
            orientation='h',
            marker=dict(
                color=sorted_data['Importance'],
                colorscale='Viridis'
            ),
            text=sorted_data['Importance'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(sorted_data) * 25),
            showlegend=False
        )
        
        return fig
    
    def create_prediction_comparison(self, actual, predicted, title="Prediction vs Actual"):
        """
        Create a prediction comparison plot
        
        Args:
            actual (pd.Series): Actual values
            predicted (pd.Series): Predicted values
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='blue',
                opacity=0.6,
                size=6
            ),
            hovertemplate='<b>Prediction vs Actual</b><br>' +
                         'Actual: %{x:.2f}<br>' +
                         'Predicted: %{y:.2f}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_residual_plot(self, actual, predicted, title="Residual Analysis"):
        """
        Create a residual analysis plot
        
        Args:
            actual (pd.Series): Actual values
            predicted (pd.Series): Predicted values
            title (str): Chart title
            
        Returns:
            plotly.graph_objects.Figure: Residual plot
        """
        residuals = actual - predicted
        
        fig = go.Figure()
        
        # Residual scatter plot
        fig.add_trace(go.Scatter(
            x=predicted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color='green',
                opacity=0.6,
                size=6
            ),
            hovertemplate='<b>Residual Analysis</b><br>' +
                         'Predicted: %{x:.2f}<br>' +
                         'Residual: %{y:.2f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            height=500,
            showlegend=True
        )
        
        return fig
