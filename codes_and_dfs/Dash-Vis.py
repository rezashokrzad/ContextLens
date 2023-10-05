# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:22:00 2022

@author: Reza Shokrzad
"""

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import flask
import os
import utils
import dash_bootstrap_components as dbc
import numpy as np


#defining empty dataframe for user inputs
user_doc = pd.DataFrame()
Label1 = pd.DataFrame()
Label2 = pd.DataFrame()

#read dataframes of adjectives
df1 = pd.read_excel('SemCor_interactive_df_present.xlsx')
df2 = pd.read_excel('SemCor_interactive_df_state171.xlsx')
df3 = pd.read_excel('SemCor_interactive_df_cold.xlsx')
df4 = pd.read_excel('SemCor_interactive_df_great.xlsx')
df5 = pd.read_excel('SemCor_interactive_df_domestic.xlsx')
df6 = pd.read_excel('Hotel_df_list1.xlsx')
df7 = pd.read_excel('Hotel_df_list2.xlsx')
df8 = pd.read_excel('Hotel_df_list3.xlsx')
df9 = pd.read_excel('Hotel_df_list4.xlsx')
df0 = pd.DataFrame()

#receive tooltip content
tooltip_columns = ['text', 'idx', 'Label1' , 'Label2', 'simple_majority_voting', 'weighted_majority_voting']
tooltip_list = df1[tooltip_columns].values.tolist()

#dash setup
external_stylesheets = [
    'https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dict_main = {'df_present':df1, 'df_state':df2, 'df_cold':df3, 'df_great':df4,
             'df_domestic':df5, 'df_hotel_list1':df6, 'df_hotel_list2':df7,
             'df_hotel_list3':df8, 'df_hotel_list4':df9, 'df_user':df0}
data = list(dict_main.keys())
data_2d = list(dict_main.keys())

dict_method = {'PCA_BERT': 'PCA_BERT', 'UMP_BERT': 'UMP_BERT'}
methods = list(dict_method.keys())

dict_label = {'Label1': 'Label1', 'Label2': 'Label2', 'simple_majority_voting': 'simple_majority_voting',
              'weighted_majority_voting': 'weighted_majority_voting','No label':'No label'}
labels = list(dict_label.keys())

dict_comp_x = {'Dim1': 'x_1', 'Dim2': 'x_2', 'Dim3': 'x_3', 'Dim4': 'x_4', 'Dim5': 'x_5','Dim6': 'x_6', 'Dim7': 'x_7', 'Dim8': 'x_8', 'Dim9': 'x_9', 'Dim10': 'x_10'}
comp_x = list(dict_comp_x.keys())

dict_comp_y = {'Dim2': 'y_2', 'Dim1': 'y_1', 'Dim3': 'y_3', 'Dim4': 'y_4', 'Dim5': 'y_5','Dim6': 'y_6', 'Dim7': 'y_7', 'Dim8': 'y_8', 'Dim9': 'y_9', 'Dim10': 'y_10'}
comp_y = list(dict_comp_y.keys())

dict_comp_z = {'Dim3': 'z_3', 'Dim1': 'z_1', 'Dim2': 'z_2', 'Dim4': 'z_4', 'Dim5': 'z_5','Dim6': 'z_6', 'Dim7': 'z_7', 'Dim8': 'z_8', 'Dim9': 'z_9', 'Dim10': 'z_10'}
comp_z = list(dict_comp_z.keys())

hovertext = "%{customdata[0]}<br>idx=%{customdata[1]}<br>Sense Label=%{customdata[2]} \
<br>Word Label=%{customdata[3]}<br>Simple Majority Voting=%{customdata[4]}<br>Weighted Majority Voting=%{customdata[5]}<extra></extra>"
clr_scale = [[0, 'rgb(255,0,0)'], [0.1, 'rgb(255,128,50)'], [0.2, 'rgb(100,205,200)'], 
             [0.3, 'rgb(0,255,0)'], [0.4, 'rgb(140,150,140)'], 
             [0.5, 'rgb(50,50,50)'], [0.6, 'rgb(255,50,255)'], 
             [0.7, 'rgb(120,190,120)'], [0.8, 'rgb(200,50,235)'],
             [0.9, 'rgb(100,30,0)'], [1, 'rgb(0,0,255)']]

#symbbols of shape datapoints in scatter3d
symbs = ['circle', 'square', 'x', 'diamond', 'cross', 'circle-open', 'square-open', 'diamond-open']
#size of symbol list
sizes = [8, 7, 9, 12, 20, 20, 20, 20]
#sctter3d setup
scatter_plot_3d = go.Scatter3d(x=df1["PCA_BERT_Dim1"],
                                y=df1["PCA_BERT_Dim2"],
                                z=df1["PCA_BERT_Dim3"],
                                customdata=tooltip_list,
                                mode='markers',
                                marker = dict(size=np.array(sizes)[df1['Label1']], 
                                              color=df1['Label1'], 
                                              symbol=np.array(symbs)[df1['Label1']],
                                              sizemode='diameter',
                                              sizeref=1
                                              ),
                                marker_color= df1['Label1'],
                                hovertemplate=hovertext,
                                #name=None,
                                showlegend=False,
                                opacity=1)
fig3d = go.Figure(data=[scatter_plot_3d], layout=go.Layout(scene = dict(xaxis = dict(title='Dim1', tickfont=dict(size=13)),
                                                             yaxis = dict(title='Dim2', tickfont=dict(size=13)),
                                                             zaxis = dict(title='Dim3', tickfont=dict(size=13)),
                                                             camera=dict(eye=dict(x=1, y=1, z=1)))))
dcc_graph = dcc.Graph(id='Main-Graph',
                      figure=fig3d.update_layout(
                          template={'data': {'pie': [{'automargin': False, 'type': 'pie'}],
                                                                           'scatter3d': [{'line': {'width': 3},
                                                                                          'marker': {'size': 9},
                                                                                          'type': 'scatter3d'}],
                                                                           'scattergeo': [{'line': {'width': 3},
                                                                                           'marker': {'size': 9},
                                                                                           'type': 'scattergeo'}],
                                                                           'scattergl': [{'line': {'width': 3},
                                                                                          'marker': {'size': 9},
                                                                                          'type': 'scattergl'}],
                                                                           'scatterpolargl': [{'line': {'width': 3},
                                                                                               'marker': {'size': 9},
                                                                                               'type': 'scatterpolargl'}],
                                                                           'scatterpolar': [{'line': {'width': 3},
                                                                                             'marker': {'size': 9},
                                                                                             'type': 'scatterpolar'}],
                                                                           'scatter': [{'line': {'width': 3},
                                                                                        'marker': {'size': 8},
                                                                                        'type': 'scatter'}],
                                                                           'scatterternary': [{'line': {'width': 3},
                                                                                               'marker': {'size': 9},
                                                                                               'type': 'scatterternary'}],
                                                                           'table': [{'cells': {'height': 30},
                                                                                      'header': {'height': 36},
                                                                                      'type': 'table'}]},
                                                                  'layout': {'font': {'size': 16},
                                                                             'xaxis': {'title': {'standoff': 10}},
                                                                             'yaxis': {'title': {'standoff': 10}}}},
                                                        height=750,
                                                        legend={'title': 'Label'},
                                                        title= 'plot_title',
                                                        plot_bgcolor= '#ecf0f1',
                                                        paper_bgcolor='white',
                                                        margin=dict(t=150, b=0, l=0, r=10),
                                                        ))

#2d graph
scatter_plot_2d = go.Scatter(x=df1["PCA_BERT_Dim1"],
                          y=df1["PCA_BERT_Dim2"],
                          customdata=tooltip_list,
                          mode='markers',
                          marker = dict(size = 8, color = df1['Label1'], symbol = df1['Label1']),
                          hovertemplate=hovertext,
                          #name='clean',
                          showlegend=False)
fig2d = go.Figure(data=[scatter_plot_2d], layout=go.Layout(scene = dict(xaxis = dict(title='Dim1', tickfont=dict(size=11)),
                                                                        yaxis = dict(title='Dim2', tickfont=dict(size=11)))))
dcc_graph_2d = dcc.Graph(id='Main-Graph-2d', figure=fig2d.update_layout(
                                                        template="presentation", 
                                                        xaxis_title_text='Dim1',
                                                        yaxis_title_text='Dim2',
                                                        height=750,
                                                        legend={'title': 'Label'},
                                                        title='plot_title',
                                                        plot_bgcolor='rgb(240,245,250)',
                                                        margin=dict(t=150, b=150, l=50, r=50)))

dcc_upload = dcc.Upload(id='upload-data', children=html.Div([html.I(className="fas fa-upload", style={'margin-right': '5px'}),
                                                             'Drag your file here or ',
                                                             html.A('Browse', style={'color': '#3498db', 'text-decoration': 'none', 'font-weight': 'bold'})]),
                        style={'width': '30%', 'height': '60px','lineHeight': '60px',
                               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                               'textAlign': 'center', 'margin': '1px 1px 1px 1px', 'backgroundColor': '#E0F7FA'},
                        # Allow multiple files to be uploaded
                        multiple=False)


common_style = {'width': '90%',
                             'height': '30px',
                             'fontSize': '16px',
                             'border': '2px solid #e1e1e1',
                             'borderRadius': '5px',
                             'padding': '5px 10px',
                             'lineHeight': '50px',
                             'borderWidth': '1px',
                             'textAlign': 'center',
                             'margin': '1px 1px 1px 1px',
                             'color': '#333',
                             'transition': 'border 0.3s'}
dcc_input_text = dcc.Input(id='word', value='', type='text', placeholder='your desire word(s), separated by ","', style=common_style)
dcc_input_number = dcc.Input(id='number', value='', type='text', placeholder='#clusters (default 5 for single word)', style=common_style)
dropdown_style = {'height': '25px',
                  'fontSize': '14px',
                  'borderRadius': '5px',
                  'color': '#333',
                  'transition': 'border 0.3s'}
# Common style for labels
label_style = {'font-weight': 'bold',
               'color': '#2c3e50',
               'font-size': '0.95em',
               'margin-left': '5px',
               'margin-bottom': '-5px',
              'margin-top': '10px'}

app.layout = html.Div([
    html.Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css"),
    html.H1('ContextLens', style={'text-align': 'center', 'font-weight': 'bold'}),
    html.Center(html.Div([
    html.H6('An embedding visualization and clustering tool', style={'display': 'inline-block','margin':'6', 'color': '#2c3e50', 'font-family': 'Comic Sans MS'}),
    html.Span(' '),
    html.I(className="fas fa-info-circle", style={'margin-right': '3px', 'color': '#3498db'}),
    html.A("(Instruction and about)", href='https://github.com/rezashokrzad/ContextLens/blob/main/ContextLens%20Instruction.md',
           target="_blank", style={'color': '#3498db', 'text-decoration': 'none', 'font-family': 'Comic Sans MS'})
    ])),
    html.Center([dcc.Loading(id="loading", type="circle", fullscreen=True,
                             children=[html.Div(dcc_upload,id='output-data-upload'),
                                       html.Div([
                     html.Div(dcc_input_text,id="input_word", style={'width': '17%', 'display': 'inline-block'}),
                     html.Div(dcc_input_number,id="input_number", style={'width': '17%', 'display': 'inline-block'})]),
                  html.Button('Process', id='process_button', style={'background-color': '#3498db', 'color': 'white', 'border': 'none', 'padding': '4px 20px', 'cursor': 'pointer'}),
                  html.Div(html.A(id='log', children='Ready', style={"color": "blue"}))

                 ])]),

    html.Div(id='output_div'),
    html.Div([
    html.Div([html.I(className="fas fa-database", style={'margin-right': '2px'}),
              html.Label("Dataframe", style=label_style),
              dcc.Dropdown(id='data-dropdown',
                           options=[{'label': label, 'value': label} for label in data],
                           value=list(dict_main.keys())[0],
                           multi=False,
                           clearable=False,
                           searchable=False, style=dropdown_style)], style={'width': '10%', 'display': 'inline-block'}),
        
    html.Div([html.I(className="fas fa-cogs", style={'margin-right': '2px'}),
              html.Label("Method", style=label_style),
              dcc.Dropdown(id='method-dropdown',
                               options=[{'label': label, 'value': label} for label in methods],
                               value=list(dict_method.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '10%', 'display': 'inline-block'}),
    html.Div([html.I(className="fas fa-palette", style={'margin-right': '2px'}),
              html.Label("Color Label", style=label_style),
              dcc.Dropdown(id='label-dropdown',
                               options=[{'label': label, 'value': label} for label in labels],
                               value=list(dict_label.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '10%', 'display': 'inline-block'}),
    html.Div([html.I(className="fas fa-shapes", style={'margin-right': '2px'}),
              html.Label("Shape Label", style=label_style),
              dcc.Dropdown(id='label-dropdown-shape',
                               options=[{'label': label, 'value': label} for label in labels],
                               value=list(dict_label.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '10%', 'display': 'inline-block'}),
        
    html.Div([html.I(className="fas fa-chart-line", style={'margin-right': '2px'}),
              html.Label("X-Axis", style=label_style),
              dcc.Dropdown(id='x-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_x],
                               value=list(dict_comp_x.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '6%', 'display': 'inline-block'}),
                                 
    html.Div([html.I(className="fas fa-chart-line", style={'margin-right': '2px'}),
              html.Label("Y-Axis", style=label_style),
              dcc.Dropdown(id='y-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_y],
                               value=list(dict_comp_y.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '6%', 'display': 'inline-block'}),
    html.Div([html.I(className="fas fa-chart-line", style={'margin-right': '2px'}),
              html.Label("Z-Axis", style=label_style),
              dcc.Dropdown(id='z-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_z],
                               value=list(dict_comp_z.keys())[0],
                               multi=False,
                               clearable=False,
                               searchable=False, style=dropdown_style)],  style={'width': '6%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),
    
    html.Div([
    html.Div(html.A(id='download-link', children='Download File Emb')),
    html.Div(html.A(id='download-link-emb', children='Download File'))
    ], style={'textAlign': 'left', 'marginTop': '20px', 'marginLeft': '400px'}),
    
    html.Center([
    html.Div(dcc_graph, style={'width': '48%', 'display': 'inline-block', 'borderRadius': '20px', 'overflow': 'hidden', 'marginRight': '0.5%'}),
    html.Div(dcc_graph_2d, style={'width': '45%', 'display': 'inline-block', 'borderRadius': '20px', 'overflow': 'hidden', 'marginLeft': '0.5%'})])
    ], style={'backgroundColor': '#f5f5f5'})


@app.callback(
    [Output('download-link', 'children'),
     Output('download-link-emb', 'children'),
     Output('Main-Graph', 'figure'),
     Output('Main-Graph-2d', 'figure')],
    [Input('data-dropdown', 'value'),
     Input('method-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('label-dropdown-shape', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('z-axis-dropdown', 'value')],
    [State('Main-Graph', 'figure'),
     State('Main-Graph-2d', 'figure')]
    )
def updateGraph(df_name, method_name, label_name, label_name_shape, x_field, y_field, z_field, data, data_2d):
    global symbs
    global sizes
    source = data['data']
    source_2d = data_2d['data']
    df = dict_main[df_name]
    source[0].update({'customdata': df[tooltip_columns].values.tolist()})
    source_2d[0].update({'customdata': df[tooltip_columns].values.tolist()})
    
    with open("data.txt", "w") as text_file:
        text_file.write(str(data['data'][0]))
    with open("layout.txt", "w") as text_file:
        text_file.write(str(data['layout']))
    
    try:
        color_label = df[label_name].tolist()
    except:
        color_label = 0
    try:
        color_shape = df[label_name_shape]
    except:
        color_shape = 0
    
    try:
        symbol_3d = np.array(symbs)[df[label_name_shape]]
    except:
        symbol_3d = 0
    
    try:
        size_symbol_3d = np.array(sizes)[df[label_name_shape]]
    except:
        size_symbol_3d = 4
    
    if x_field and y_field and z_field:
        source[0].update({'x': df[method_name+'_'+x_field].tolist(),
                          'y': df[method_name+'_'+y_field].tolist(),
                          'z': df[method_name+'_'+z_field].tolist()})
        source_2d[0].update({'x': df[method_name+'_'+x_field].tolist(),
                             'y': df[method_name+'_'+y_field].tolist()})
        source[0].update({'marker':{'color':color_label,
                                    'size':size_symbol_3d,
                                    'colorscale':clr_scale,
                                    'symbol':symbol_3d,
                                    'sizemode':'diameter',
                                    'sizeref':1
                                    }})

        source_2d[0].update({'marker':{'color': color_label,
                                       'size':8, 'colorscale':clr_scale,
                                       'symbol':color_shape
                                       }})
        
        data['layout'].update({'title': {'text': str(method_name)+" | "+x_field+' vs '+y_field+' vs '+z_field,
                                         'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': 'black',  'weight': 'bold'},
                                         'x': 0.5,  'y': 0.95,  'xanchor': 'center', 'yanchor': 'top'}})
        data['layout']['scene']['xaxis'] = {**data['layout']['scene']['xaxis'], 'title': {'text': x_field}}
        data['layout']['scene']['yaxis'] = {**data['layout']['scene']['yaxis'], 'title': {'text': y_field}}
        data['layout']['scene']['zaxis'] = {**data['layout']['scene']['zaxis'], 'title': {'text': z_field}}

        data_2d['layout'].update({'title': {'text': str(method_name)+" | "+x_field+' vs '+y_field,
                                            'font': {'family': 'Arial, sans-serif', 'size': 18, 'color': 'black',  'weight': 'bold'},
                                            'x': 0.5,  'y': 0.95,  'xanchor': 'center', 'yanchor': 'top'}})
        data_2d['layout']['xaxis'] = {**data_2d['layout']['xaxis'], 'title': {'text': x_field}}
        data_2d['layout']['yaxis'] = {**data_2d['layout']['yaxis'], 'title': {'text': y_field}}
        
    current_df_emb = dict_main[df_name]
    current_df_name_emb =  df_name + '_emb_' + str(np.random.randint(100000)+1)
    current_df_emb.to_excel("./download/" + current_df_name_emb +".xlsx", encoding='utf-8')
    current_df = current_df_emb.copy()
    
    for i in range(10):
        current_df.drop(['PCA_BERT_Dim'+str(i+1)],inplace=True,axis=1)
        current_df.drop(['UMP_BERT_Dim'+str(i+1)],inplace=True,axis=1)
 
    current_df_name =  df_name + '_' + str(np.random.randint(100000)+1)
    current_df.to_excel("./download/" + current_df_name +".xlsx", encoding='utf-8')
    
    url_string_emb = './download/' +current_df_name_emb+ '.xlsx'
    url_string = './download/' +current_df_name+ '.xlsx'
    
    return [[html.I(className="fas fa-cloud-download-alt", style={'margin-right': '5px', 'color': '#3498db'}),
             html.A("Download dataframe with embeddings", href=url_string_emb, target="_blank", download=url_string_emb
                    , style={'color': '#3498db', 'text-decoration': 'none', 'font-family': 'Comic Sans MS'})],
            [html.I(className="fas fa-cloud-download-alt", style={'margin-right': '5px', 'color': '#3498db'}),
             html.A("Download dataframe without embeddings", href=url_string, target="_blank", download=url_string,
                    style={'color': '#3498db', 'text-decoration': 'none', 'font-family': 'Comic Sans MS'})],
            {'data': source, 'layout': data['layout']},
            {'data': source_2d, 'layout': data_2d['layout']}]

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content, filename, date):
    global user_doc
    global Label1
    global Label2
    df = utils.parse_contents(content, filename, date)
    
    outputlog = 'Upload done'
    
    Label1 = pd.DataFrame()
    Label2 = pd.DataFrame()
    
    try:
        if not df.empty:
            print('df shape:',df.shape)
            if df.shape[1] == 1:
                user_doc = df
            elif df.shape[1] == 2:
                user_doc = df.iloc[:,0].to_frame()
                Label1 =  df.iloc[:,1]
            elif df.shape[1] == 3:
                user_doc = df.iloc[:,0].to_frame()
                Label1 =  df.iloc[:,1]
                Label2 =  df.iloc[:,2]
            else:
                raise TypeError('Too many columns')
    except:
        outputlog = 'Upload failed'
        print("except")
   
    return [html.Div(dcc_upload, id='output-data-upload'),
            # outputlog
            ]

@app.callback(
               Output('log', 'children'),
              [Input('process_button', 'n_clicks')],
              [State('word', 'value'),
               State('number', 'value')],
          )
def process_file(clicks, input_w, input_n):
    global user_doc
    global Label1
    global Label2
    input_w = [word.strip() for word in input_w.split(',') if word.strip()]
    
    try:
        input_n = int(input_n)
        if input_n <= 0:
            return html.A("Number of clusters should be a positive integer", id="log")    
    except ValueError:
        if input_n == '':
            input_n = 5
        else:
            return html.A("Invalid number of clusters", id="log")    

    if not user_doc.empty and user_doc.shape[0] > 200:
        return html.A("Number of sentences should not exceed 200", id="log", style={'color': 'red', 'font-family': 'Comic Sans MS'})
    elif len(input_w) == 1 and len(input_w[0]) != 0 and not user_doc.empty:
        try:
            df0 = utils.get_dataframe(user_doc, input_w,Label1,Label2, input_n) # sense level
            dict_main['df_user'] = df0
        except:
            return html.A("The provided word is not presented in your doc", id="log", style={'color': 'red', 'font-family': 'Comic Sans MS'})
            
    elif not input_w or len(input_w[0]) == 0 and user_doc.empty:
        return html.A("Please upload your file and provide target words", id="log", style={'color': '#3498db', 'font-family': 'Comic Sans MS'})
    elif len(input_w[0]) == 0:
        return html.A("Please enter words correctly", id="log", style={'color': 'red', 'font-family': 'Comic Sans MS'})
        # return "Please enter words correctly"
    elif not user_doc.empty:
        df0 = utils.get_dataframe(user_doc, input_w,Label1,Label2, len(input_w)) # word level
        dict_main['df_user'] = df0
    else:
        return html.A("Something is wrong with inputs", id="log", style={'color': 'red'})
    
    return html.A("Process done, please select df_user from dataframe dropdown", id="log", style={'color': 'green', 'font-family': 'Comic Sans MS'})

@app.server.route("/download/<path:path>")
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'download'), path)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, host="0.0.0.0", port=80)
