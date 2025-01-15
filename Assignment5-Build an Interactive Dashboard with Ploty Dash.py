#TASK 1:Add a Launch Site Drop-down Input Component
# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px


# Read the SpaceX dataset
spacex_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("SpaceX Launch Records Dashboard", style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    # TASK 1: Add a Launch Site Dropdown
    html.Div([
        html.Label("Select Launch Site:", style={'font-size': 20}),
        dcc.Dropdown(
    id='site-dropdown',
    options=[{'label': 'All Sites', 'value': 'ALL'}] + [
        {'label': site, 'value': site} for site in spacex_df['Launch Site'].unique()
    ],
    value='ALL',
    placeholder="Select a Launch Site",
    searchable=True,
    style={'width': '80%', 'margin': 'auto'}
),
    ]),
    
    html.Br(),
    
    # TASK 2: Add a pie chart for success rates
    html.Div(dcc.Graph(id='success-pie-chart')),
    
    html.Br(),
    
    # TASK 3: Add a Range Slider for Payload Mass
    html.Div([
        html.Label("Select Payload Range (Kg):", style={'font-size': 20}),
        dcc.RangeSlider(
            id='payload-slider',
            min=0,
            max=10000,
            step=1000,
            marks={i: f'{i} Kg' for i in range(0, 10001, 1000)},
            value=[min_payload, max_payload],
        ),
    ], style={'margin': 'auto', 'width': '80%'}),
    
    html.Br(),
    
    # Task 4: Scatter plot for payload vs. outcome
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])
  

@app.callback(
    Output('success-pie-chart', 'figure'),
    Input('site-dropdown', 'value')
)
def get_pie_chart(selected_site):
    # Check if the selected site is valid
    if selected_site == 'ALL':
        # Group the entire dataframe by success/failure
        pie_data = spacex_df.groupby('class').size().reset_index(name='count')
        fig = px.pie(
            pie_data,
            values='count',
            names='class',
            title='Total Success vs Failure for All Sites'
        )
    else:
        # Filter the dataframe for the selected site
        site_data = spacex_df[spacex_df['Launch Site'] == selected_site]
        if not site_data.empty:
            pie_data = site_data.groupby('class').size().reset_index(name='count')
            fig = px.pie(
                pie_data,
                values='count',
                names='class',
                title=f'Success vs Failure for {selected_site}'
            )
        else:
            # Handle case where no data is available for the selected site
            fig = px.pie(
                names=['No Data'],
                values=[1],
                title=f'No Data for {selected_site}'
            )
    return fig


# Callback for Task 4: Update scatter plot based on selected site and payload range
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_scatter_plot(entered_site, payload_range):
    filtered_df = spacex_df[
        (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
        (spacex_df['Payload Mass (kg)'] <= payload_range[1])
    ]
    if entered_site == 'ALL':
        fig = px.scatter(
            filtered_df, x='Payload Mass (kg)', y='class',
            color='Booster Version Category',
            title='Payload vs. Outcome for All Sites',
            labels={'class': 'Launch Outcome'}
        )
    else:
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        fig = px.scatter(
            filtered_df, x='Payload Mass (kg)', y='class',
            color='Booster Version Category',
            title=f"Payload vs. Outcome for site {entered_site}",
            labels={'class': 'Launch Outcome'}
        )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)