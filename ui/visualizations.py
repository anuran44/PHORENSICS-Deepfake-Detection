import numpy as np
import plotly.graph_objects as go

def create_pure_heatmap(z_grid):
    fig = go.Figure(data=go.Heatmap(z=z_grid, colorscale='Reds', zmin=0, zmax=4.5, hovertemplate='Anomaly Score: %{z:.2f}<extra></extra>'))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=350, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_spectral_forensics_plot(r_vals, p_vals, alpha, intercept, is_fake):
    limit = min(200, len(r_vals))
    x_freq = r_vals[1:limit] 
    y_actual_log = np.log10(p_vals[1:limit] + 1e-10)
    log_x = np.log10(x_freq)

    raw_alpha = alpha / 0.70
    y_calibrated_log = y_actual_log + (raw_alpha - alpha) * log_x

    y_upper_threat = -2.0 * log_x + intercept
    y_lower_threat = -3.5 * log_x + intercept
    
    y_max = intercept + 1.0
    y_min = np.min(y_calibrated_log) - 1.5

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_freq, y=y_upper_threat, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_freq, y=y_lower_threat, mode='lines', fill='tonexty', fillcolor='rgba(211, 47, 47, 0.08)', line=dict(color='rgba(211,47,47,0.8)', width=1.5, dash='dash'), name='Threat Zone Boundary'))
    fig.add_trace(go.Scatter(x=x_freq, y=y_upper_threat, mode='lines', line=dict(color='rgba(211,47,47,0.8)', width=1.5, dash='dash'), showlegend=False, hoverinfo='skip'))

    line_color = '#d32f2f' if is_fake else '#1565c0'
    fig.add_trace(go.Scatter(x=x_freq, y=y_calibrated_log, mode='lines', name=f'Calibrated Signal (α = {alpha:.2f})', line=dict(color=line_color, width=3)))

    fig.add_annotation(x=np.log10(15), y=-2.75 * np.log10(15) + intercept, text="THREAT ZONE<br>(2.0 ≤ α ≤ 3.5)", showarrow=False, font=dict(color="#c62828", size=11, family="Arial Black"))
    fig.add_annotation(x=np.log10(15), y=-1.0 * np.log10(15) + intercept, text="SAFE ZONE<br>(α < 2.0)", showarrow=False, font=dict(color="#2e7d32", size=11, family="Arial Black"))
    fig.add_annotation(x=np.log10(15), y=-4.2 * np.log10(15) + intercept, text="SAFE ZONE<br>(α > 3.5)", showarrow=False, font=dict(color="#2e7d32", size=11, family="Arial Black"))

    fig.update_layout(
        xaxis_type="log", height=400, margin=dict(l=30, r=20, t=30, b=30),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Log10 Spatial Frequency",
        yaxis_title="Log10 Signal Energy (Power Spectrum)",
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', range=[y_min, y_max])
    )
    return fig

def create_speedometer(value, title, max_val=100, is_prob=True):
    color = "#fa4d56" if (value > 50 and is_prob) or (value > 3.5 and not is_prob) else "#24a148"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': "%" if is_prob else " Z", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': color},
            'steps': [{'range': [0, max_val/2], 'color': "rgba(128,128,128,0.05)"}, {'range': [max_val/2, max_val], 'color': "rgba(128,128,128,0.15)"}],
            'threshold': {'line': {'color': "#fa4d56", 'width': 3}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_z_distribution_plot(z_array, color):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=z_array, marker_color=color, nbinsx=60, name="Patch Z-Scores"))
    fig.add_vline(x=3.5, line_dash="dash", line_color="#fa4d56", annotation_text="Threat (3.5)")
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=20, b=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_rgb_matrix_plot(matrix):
    labels = ['Red', 'Green', 'Blue']
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=labels, colorscale='Blues', zmin=0, zmax=1, 
        text=np.round(matrix, 2), texttemplate="%{text}",
        hovertemplate="Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20), title={'text': "RGB Matrix", 'font': {'size': 14}}, paper_bgcolor='rgba(0,0,0,0)')
    return fig