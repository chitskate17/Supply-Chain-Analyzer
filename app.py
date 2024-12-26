import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from predict import SupplyChainPredictionSystem

# Initialize
app = dash.Dash(__name__, suppress_callback_exceptions=True)
df = pd.read_csv('updated_supply_chain_data.csv')
prediction_system = SupplyChainPredictionSystem('updated_supply_chain_data.csv')
prediction_system.train()


def update_graph_layout(fig, title):
    fig.update_layout(
        title=title,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Arial',
        title_x=0.5,
        title_font=dict(size=16, color='#2c3e50'),
        xaxis=dict(showgrid=True, gridcolor='#f2f2f2'),
        yaxis=dict(showgrid=True, gridcolor='#f2f2f2'),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


app.layout = html.Div([
    # Header and filters remain the same
    html.Div([
        html.H1('SUPPLY CHAIN ANALYSIS'),
        html.Div([
            dcc.Dropdown(id='product-type-filter',
                         options=[{'label': x, 'value': x} for x in df['Product type'].unique()], value='all'),
            dcc.Dropdown(id='location-filter', options=[{'label': x, 'value': x} for x in df['Location'].unique()],
                         value='all'),
            dcc.Dropdown(id='supplier-filter', options=[{'label': x, 'value': x} for x in df['Supplier name'].unique()],
                         value='all'),
        ])
    ]),
    dcc.Tabs(id='tabs', value='inventory', children=[
        dcc.Tab(label='Inventory Overview', value='inventory'),
        dcc.Tab(label='Supplier & Shipping', value='shipping'),
        dcc.Tab(label='Production Analysis', value='production'),
        dcc.Tab(label='Quality Control', value='quality'),
        dcc.Tab(label='Predictive Analysis', value='predictive')
    ]),
    html.Div(id='tab-content')
])


@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('product-type-filter', 'value'),
     Input('location-filter', 'value'),
     Input('supplier-filter', 'value')]
)
def render_tab_content(tab, product_type, location, supplier):
    filtered_df = df.copy()
    if product_type != 'all':
        filtered_df = filtered_df[filtered_df['Product type'] == product_type]
    if location != 'all':
        filtered_df = filtered_df[filtered_df['Location'] == location]
    if supplier != 'all':
        filtered_df = filtered_df[filtered_df['Supplier name'] == supplier]

    if filtered_df.empty:
        return html.Div("No data available for the selected filters.", style={'color': 'red'})

    if tab == 'inventory':
        return render_inventory_tab(filtered_df)
    elif tab == 'shipping':
        return render_shipping_tab(filtered_df)
    elif tab == 'production':
        return render_production_tab(filtered_df)
    elif tab == 'quality':
        return render_quality_tab(filtered_df)
    elif tab == 'predictive':
        return render_predictive_tab()


def render_inventory_tab(filtered_df):
    # Existing inventory tab code remains the same
    total_stock = filtered_df['Stock levels'].sum()
    total_revenue = filtered_df['Revenue generated'].sum()

    kpi_cards = html.Div([
        html.Div([html.H3("Total Stock"), html.H2(f"{total_stock:,.0f}")], className='kpi-card'),
        html.Div([html.H3("Total Revenue"), html.H2(f"${total_revenue:,.2f}")], className='kpi-card')
    ])

    stock_fig = px.bar(
        filtered_df.groupby('Product type')['Stock levels'].sum().reset_index(),
        x='Product type',
        y='Stock levels',
        title="Stock Levels by Product Type"
    )
    stock_fig = update_graph_layout(stock_fig, "Stock Levels by Product Type")

    return html.Div([kpi_cards, dcc.Graph(figure=stock_fig)])


def render_shipping_tab(filtered_df):
    if 'Delivery time' not in filtered_df.columns:
        return html.Div("Delivery time data not available", style={'color': 'red'})

    shipping_analysis = html.Div([
        html.Div([
            dcc.Graph(figure=px.pie(filtered_df,
                                    names='Shipping carriers',
                                    title='Distribution of Shipping Carriers')),
            dcc.Graph(figure=px.box(filtered_df,
                                    x='Shipping carriers',
                                    y='Delivery time',  # Changed from Lead times
                                    title='Delivery Time by Carrier'))
        ], style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'gap': '20px'}),
        dcc.Graph(figure=px.scatter(filtered_df,
                                    x='Shipping costs',  # Changed from Distance
                                    y='Delivery time',  # Changed from Lead times
                                    color='Shipping carriers',
                                    title='Shipping Cost vs Delivery Time by Carrier'))
    ])
    return shipping_analysis


def render_production_tab(filtered_df):
    if 'Production volumes' not in filtered_df.columns:
        return html.Div("Production data not available", style={'color': 'red'})

    production_analysis = html.Div([
        html.Div([
            dcc.Graph(figure=px.line(filtered_df,
                                     x='Order date',
                                     y='Production volumes',
                                     title='Production Volume Over Time')),
            dcc.Graph(figure=px.bar(filtered_df.groupby('Product type')['Manufacturing costs'].mean().reset_index(),
                                    # Changed from Production costs
                                    x='Product type',
                                    y='Manufacturing costs',
                                    title='Average Manufacturing Cost by Product'))
        ]),
        dcc.Graph(figure=px.scatter(filtered_df,
                                    x='Production volumes',  # Changed from Production volumes
                                    y='Manufacturing costs',  # Changed from Production costs
                                    color='Product type',
                                    title='Production Volume vs Manufacturing Costs'))
    ])
    return production_analysis


def render_quality_tab(filtered_df):
    if 'Defect rates' not in filtered_df.columns:
        return html.Div("Quality data not available", style={'color': 'red'})

    quality_analysis = html.Div([
        html.Div([
            dcc.Graph(figure=px.bar(filtered_df.groupby('Product type')['Defect rates'].mean().reset_index(),
                                    x='Product type',
                                    y='Defect rates',
                                    title='Defect Rates by Product')),
            dcc.Graph(figure=px.scatter(filtered_df,
                                        x='Number of products sold',
                                        y='Defect rates',
                                        color='Product type',
                                        title='Sales Volume vs Defect Rates'))
        ]),
        dash_table.DataTable(
            data=filtered_df.sort_values('Defect rates', ascending=False).head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in ['Product type', 'Defect rates']],  # Removed Quality metrics
            style_table={'overflowX': 'auto'},
        )
    ])
    return quality_analysis


def render_predictive_tab():
    styles = {
        'container': {
            'max-width': '800px',
            'margin': '0 auto',
            'padding': '2rem'
        },
        'form': {
            'background': 'white',
            'padding': '1.5rem',
            'border-radius': '0.5rem',
            'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
            'width': '100%',
            'max-width': '400px',
            'margin-left': '12rem'
        },
        'label': {
            'display': 'block',
            'margin-bottom': '0.5rem',
            'font-weight': '500',
            'color': '#374151'
        },
        'input': {
            'width': '100%',
            'padding': '0.5rem',
            'border': '1px solid #D1D5DB',
            'border-radius': '0.375rem',
            'margin-bottom': '1rem'
        },
        'button': {
            'width': '100%',
            'background': '#2563EB',
            'color': 'white',
            'padding': '0.75rem',
            'border-radius': '0.375rem',
            'font-weight': '500',
            'cursor': 'pointer'
        },
        'prediction_container': {
            'background': 'white',
            'padding': '1.5rem',
            'border-radius': '0.5rem',
            'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
            'margin-top': '1rem',
            'width': '100%',
            'max-width': '400px',
            'margin-left': '12rem'
        },
        'prediction_item': {
            'padding': '0.75rem',
            'border-bottom': '1px solid #E5E7EB',
            'display': 'flex',
            'justify-content': 'space-between'
        },
        'prediction_label': {
            'color': '#374151',
            'font-weight': '500'
        },
        'prediction_value': {
            'color': '#2563EB',
            'font-weight': '500'
        },
        'title': {
            'text-align': 'center',
            'margin-bottom': '2rem',
            'color': '#1F2937',
            'font-size': '1.5rem',
            'font-weight': '600'
        }
    }

    return html.Div([
        html.H3("Supply Chain Prediction Tool", style=styles['title']),
        html.Div([
            html.Div([
                html.Label("Product Type", style=styles['label']),
                dcc.Dropdown(
                    id='pred-product-type',
                    options=[{'label': x, 'value': x} for x in df['Product type'].unique()],
                    style=styles['input']
                ),
                html.Label("Price", style=styles['label']),
                dcc.Input(
                    id='pred-price',
                    type='number',
                    placeholder="Enter price",
                    style=styles['input']
                ),
                html.Label("Availability", style=styles['label']),
                dcc.Input(
                    id='pred-availability',
                    type='number',
                    placeholder="Enter availability",
                    style=styles['input']
                ),
                html.Label("Customer Demographics", style=styles['label']),
                dcc.Dropdown(
                    id='pred-demographics',
                    options=[{'label': x, 'value': x} for x in df['Customer demographics'].unique()],
                    style=styles['input']
                ),
                html.Label("Location", style=styles['label']),
                dcc.Dropdown(
                    id='pred-location',
                    options=[{'label': x, 'value': x} for x in df['Location'].unique()],
                    style=styles['input']
                ),
                html.Button(
                    'Predict',
                    id='predict-button',
                    n_clicks=0,
                    style=styles['button']
                ),
            ], style=styles['form']),
            html.Div(id='prediction-output', style=styles['prediction_container'])
        ], style=styles['container'])
    ])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('pred-product-type', 'value'),
     State('pred-price', 'value'),
     State('pred-availability', 'value'),
     State('pred-demographics', 'value'),
     State('pred-location', 'value')]
)
def update_prediction(n_clicks, product_type, price, availability, demographics, location):
    if n_clicks == 0:
        return html.Div()

    if not all([product_type, price, availability, demographics, location]):
        return html.Div("Please fill in all fields", style={
            'color': 'red',
            'text-align': 'center',
            'padding': '1rem'
        })

    features = {
        'Product type': product_type,
        'Price': price,
        'Availability': availability,
        'Customer demographics': demographics,
        'Location': location
    }

    predictions = prediction_system.predict(features)

    prediction_item_style = {
        'padding': '0.75rem',
        'border-bottom': '1px solid #E5E7EB',
        'display': 'flex',
        'justify-content': 'space-between'
    }

    return html.Div([
        html.H4("Prediction Results", style={
            'color': '#1F2937',
            'font-size': '1.25rem',
            'font-weight': '600',
            'margin-bottom': '1rem',
            'text-align': 'center'
        }),
        html.Div([
            html.Div([
                html.Span("Expected Sales", style={'color': '#374151', 'font-weight': '500'}),
                html.Span(f"{predictions['Number of products sold']}", style={'color': '#2563EB', 'font-weight': '500'})
            ], style=prediction_item_style),
            html.Div([
                html.Span("Projected Revenue", style={'color': '#374151', 'font-weight': '500'}),
                html.Span(f"₹{predictions['Revenue generated']:,.2f}", style={'color': '#2563EB', 'font-weight': '500'})
            ], style=prediction_item_style),
            html.Div([
                html.Span("Recommended Carrier", style={'color': '#374151', 'font-weight': '500'}),
                html.Span(f"{predictions['Best shipping carrier']}", style={'color': '#2563EB', 'font-weight': '500'})
            ], style=prediction_item_style),
            html.Div([
                html.Span("Suggested Transport", style={'color': '#374151', 'font-weight': '500'}),
                html.Span(f"{predictions['Best transportation mode']}", style={'color': '#2563EB', 'font-weight': '500'})
            ], style=prediction_item_style),
            html.Div([
                html.Span("Optimal Route", style={'color': '#374151', 'font-weight': '500'}),
                html.Span(f"{predictions['Best route']}", style={'color': '#2563EB', 'font-weight': '500'})
            ], style=prediction_item_style)
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)