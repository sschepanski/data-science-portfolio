import plotly.graph_objs as go
import plotly.express as px

def plot_series_with_events(series_id, train_series, train_events, enmo=False, anglez=True):
    '''
    This function takes a series_id and the train_series and train_events DataFrames and 
    plots the anglez and/or the enmo values over time with the event flags and night numbers.
    '''
    # Filter the DataFrame based on the series_id
    sample_serie = train_series[train_series['series_id'] == series_id]
    
    # Filter event data based on the series_id
    sample_events = train_events[train_events['series_id'] == series_id]
    sample_onset = sample_events.loc[sample_events['event'] == 'onset', ['timestamp','night']].dropna().reset_index(drop=True)
    sample_wakeup = sample_events.loc[sample_events['event'] == 'wakeup', 'timestamp'].dropna().reset_index(drop=True)
    
    # Helper function to plot data and events
    def plot_data_and_events(data, ylabel, show_figure=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_serie['timestamp'], y=sample_serie[data], name=data, line=dict(color='grey', width=.5)))

        for index, row in sample_onset.iterrows():
            onset = row['timestamp']
            wakeup = sample_wakeup.iloc[index]
            fig.add_shape(type='line', x0=onset, y0=sample_serie[data].min(), x1=onset, y1=sample_serie[data].max(), line=dict(color='red', dash='dash'), name='onset')
            fig.add_shape(type='line', x0=wakeup, y0=sample_serie[data].min(), x1=wakeup, y1=sample_serie[data].max(), line=dict(color='green', dash='dash'), name='wakeup')
            midpoint = onset + (wakeup - onset)/2
            fig.add_annotation(x=midpoint, y=sample_serie[data].max(), text=row['night'], showarrow=False, font=dict(size=10), textangle=-90)
            
        fig.update_layout(title=f'{ylabel} over Time with Event Flags - '+series_id, xaxis_title='Timestamp', yaxis_title=ylabel, showlegend=False)
        
        if show_figure:
            fig.show()

        return fig  # return the figure

    # Plot enmo and anglez
    enmo_fig = plot_data_and_events('enmo', 'ENMO Value', show_figure=enmo) if enmo else None
    anglez_fig = plot_data_and_events('anglez', 'anglez Value', show_figure=anglez) if anglez else None

    return enmo_fig if enmo else anglez_fig  # Return the figure that was plotted
