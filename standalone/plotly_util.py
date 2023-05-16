import plotly
import plotly.graph_objects as go
import numpy as np

def add_plotly_errorbar(fig, x, y, err, color=None, alpha=0.2, mode="markers+lines",
                        name='', legend_group=None, subplot_kwargs={}, **kwargs):
    if legend_group is None:
        legend_group = f'group_{name}'
        
    valid_y = ~np.isnan(y)
    y = y[valid_y]
    x = x[valid_y]
    err = err[valid_y]
    err[np.isnan(err)] = 0
    
    if color is None:
        color = kwargs.pop('marker_color')
    
    fig.add_trace(go.Scattergl(    
        x=x, 
        y=y, 
        # error_y=dict(type='data',
        #             symmetric=True,
        #             array=tuning_sem),
        name=name,
        legendgroup=legend_group,
        mode=mode,        
        marker_color=color,
        opacity=1,
        **kwargs,
        ), **subplot_kwargs)
    
    fig.add_trace(go.Scatter(
            # name='Upper Bound',
            x=x,
            y=y + err,
            mode='lines',
            marker=dict(color=color),
            line=dict(width=0),
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip',
        ), **subplot_kwargs)
    
    if 'rgb' in color:  # if starting with rgb and already have alpha, just override the alpha
        error_band_fill_color = f'{",".join(color.split(",")[:-1])}, 0.4)'
    else:  # in text ('red', 'blue', etc.)
        error_band_fill_color = f'rgba({plotly.colors.convert_colors_to_same_type(color, "rgb")[0][0].split("(")[-1][:-1]}, {alpha})'
    
    fig.add_trace(go.Scatter(
                    # name='Upper Bound',
                    x=x,
                    y=y - err,
                    mode='lines',
                    marker=dict(color=color),
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=error_band_fill_color,
                    legendgroup=legend_group,
                    showlegend=False,
                    hoverinfo='skip'
                ), **subplot_kwargs)  
