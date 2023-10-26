import pandas as pd
import yfinance as yf
import numpy as np
from dash import html
from dash import dcc, Input, Output, Dash, dash
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template




# stock = input("Please enter your ticker: ")

# def get_stock(ticker):
#     url = f'https://finance.yahoo.com/quote/{ticker}/history?p={ticker}'

#     r = requests.get(url, headers= {
#         'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

#     stockie = pd.read_html(r.text)
    
#     print(stockie)  
# get_stock(stock)

load_figure_template("cyborg")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

def stock_prices_graph(df):
      
      close_fig = go.Figure([go.Scatter( x = df['Date'], y = df['Close'],
                     line = dict(color = 'firebrick', width = 3), name = 'Close')
                     ])
      close_fig.update_layout(title = 'Close Price over time',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Close Prices'
                      )
      
      open_fig = go.Figure([go.Scatter(x=df['Date'], y= df['Open'],
                                        line = dict(color = 'green', width = 3), name = 'Open')
                                        ])
      open_fig.update_layout(title = 'Open Price over time', 
                             xaxis_title = 'Date',
                             yaxis_title = 'Open Price',
                             )


      
      return close_fig, open_fig  

def get_stock_yf(ticker, period):
     stock_history = yf.download(tickers=ticker,
                                    period=f'{period}',
                                    interval = "1d")
     stock_history.reset_index(inplace = True)
     
     stock_history['Date'] = pd.to_datetime(stock_history['Date'])
     
     stock_history['Date'] = stock_history['Date'].values.astype('datetime64[D]')
     
     stock_history.to_csv('stock_history.csv')
     return (stock_history)


def read_csv_history ():
     data = pd.read_csv('stock_history.csv')
     data.reset_index(inplace = True)
     
     data['Date'] = pd.to_datetime(data['Date'])
     
     data['Date'] = data['Date'].values.astype('datetime64[D]')
     
     return data


def predict_value(data):
     data['Date'] = data['Date'].values.astype('float64')
     X = np.array(data['Date']).reshape(-1,1)
     y= data['Close']
     reg = LinearRegression().fit(X,y)

     print(reg.predict(X))
     return reg.predict(X)



app.layout = dbc.Container([

     
          dbc.Row([
               dbc.Col([
                    html.H1("Stock Tracker", style ={'textAlign' : 'center'})
                    ], width=12)
               ]
               ),
               
          dbc.Row([
               dbc.Col([
                    html.Label('Stock ticker'),
                    dcc.Input(id = 'input', value = 'BTC-USD', type = 'text')], width = 6),
               dbc.Col([
                    html.Label('year range'),
                    dcc.Dropdown(id = 'period', options = ['5y', '1y', '2y'], value = ['5y', '1y', '2y'])], width = 6)
          ]),

          html.Br(),
          dbc.Row([
               dbc.Col([
                    html.Label('Open Amount Over time'),
                    dcc.Graph(id='output-graph', figure = {}),
               ], width=6),
          
                dbc.Col([
                    html.Label('Close Amount Over Time'),
                    dcc.Graph(id='output-graph1', figure = {})
               ], width = 6)
                    
          ])
     ])


@app.callback(
     [Output( component_id ='output-graph', component_property= 'figure'),
     Output( component_id ='output-graph1', component_property='figure')],
     [Input(component_id ='input', component_property='value'),
     Input(component_id = 'period', component_property = 'value')]

)

def update_ticker(input_value, period_value):
     data = get_stock_yf(input_value, period_value)
     
     close_fig, open_fig = stock_prices_graph(data)
     
     return  open_fig, close_fig
 




if __name__ == '__main__': 
    app.run_server()



     