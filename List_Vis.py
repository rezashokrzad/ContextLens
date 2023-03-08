import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import flask
import os

#read dataframes of adjectives
df1 = pd.read_excel('Hotel_df_list1.xlsx')
df2 = pd.read_excel('Hotel_df_list2.xlsx')
df3 = pd.read_excel('Hotel_df_list3.xlsx')

#receive tooltip content
tooltip_columns = ['text', 'idx', 'Label']
tooltip_list = df1[tooltip_columns].values.tolist()

#dash setup
app = dash.Dash()


dict_main = {f'df_list{i}': df for i, df in enumerate([df1, df2, df3], start=1)}
data = list(dict_main.keys())
data_2d = list(dict_main.keys())


dict_method = {'PCA_BERT': 'PCA_BERT', 'UMP_BERT': 'UMP_BERT'}
methods = list(dict_method.keys())


dict_label = {'Label': 'Label', 'simple_majority_voting': 'simple_majority_voting',
              'weighted_majority_voting': 'weighted_majority_voting'}
labels = list(dict_label.keys())

dict_comp_x = {'Dim1': 'x_1', 'Dim2': 'x_2', 'Dim3': 'x_3', 'Dim4': 'x_4', 'Dim5': 'x_5'}
comp_x = list(dict_comp_x.keys())

dict_comp_y = {'Dim2': 'y_2', 'Dim1': 'y_1', 'Dim3': 'y_3', 'Dim4': 'y_4', 'Dim5': 'y_5'}
comp_y = list(dict_comp_y.keys())

dict_comp_z = {'Dim3': 'z_3', 'Dim1': 'z_1', 'Dim2': 'z_2', 'Dim4': 'z_4', 'Dim5': 'z_5'}
comp_z = list(dict_comp_z.keys())

hovertext = "%{customdata[0]}<br>idx=%{customdata[1]}<br>Label=%{customdata[2]}<extra></extra>"
clr_scale = [[0, 'rgb(255,0,0)'], [0.1, 'rgb(255,128,0)'], [0.2, 'rgb(0,255,0)'], 
             [0.3, 'rgb(0,255,0)'], [0.4, 'rgb(0,255,0)'], 
             [0.5, 'rgb(50,50,50)'], [0.6, 'rgb(255,0,255)'], 
             [0.7, 'rgb(255,0,255)'], [0.8, 'rgb(255,0,255)'],
             [0.9, 'rgb(0,0,0)'], [1, 'rgb(0,0,255)']]
scatter_plot = go.Scatter3d(x=df1["PCA_BERT_Dim1"],
                            y=df1["PCA_BERT_Dim2"],
                            z=df1["PCA_BERT_Dim3"],
                            customdata=tooltip_list,
                            mode='markers',
                            marker={'size':4, 'colorscale':clr_scale},
                            marker_color= df1['Label'],
                            hovertemplate=hovertext,
                            name='clean',
                            showlegend=False,
                            opacity=0.6)
fig = go.Figure(data=[scatter_plot], layout=go.Layout(scene = dict(xaxis = dict(title='Dim1'),
                                                         yaxis = dict(title='Dim2'),
                                                         zaxis = dict(title='Dim3'))))
dcc_graph = dcc.Graph(id='Main-Graph', figure=fig.update_layout(
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
                                                                                        'marker': {'size': 9},
                                                                                        'type': 'scatter'}],
                                                                           'scatterternary': [{'line': {'width': 3},
                                                                                               'marker': {'size': 9},
                                                                                               'type': 'scatterternary'}],
                                                                           'table': [{'cells': {'height': 30},
                                                                                      'header': {'height': 36},
                                                                                      'type': 'table'}]},
                                                                  'layout': {'font': {'size': 14},
                                                                             'xaxis': {'title': {'standoff': 15}},
                                                                             'yaxis': {'title': {'standoff': 15}}}},
                                                        height=750,
                                                        legend={'title': 'Label'},
                                                        title='plot_title',
                                                        plot_bgcolor='rgb(240,240,240)',
                                                        margin=dict(t=100, b=100, l=50, r=50),
                                                        ))

#2d graph
scatter_plot_2d = go.Scatter(x=df1["PCA_BERT_Dim1"],
                          y=df1["PCA_BERT_Dim2"],
                          customdata=tooltip_list,
                          mode='markers',
                          marker={'size':10, 'colorscale':clr_scale},
                          marker_color= df1['Label'],
                          hovertemplate=hovertext,
                          name='clean',
                          showlegend=True)
dcc_graph_2d = dcc.Graph(id='Main-Graph-2d',
                      figure=go.Figure(data=[scatter_plot_2d], layout={'showlegend': False}
                                       ).update_layout(
                                                        template="presentation", 
                                                        xaxis_title_text='Dim1',
                                                        yaxis_title_text='Dim2',
                                                        height=750,
                                                        legend={'title': 'Label'},
                                                        title='plot_title',
                                                        plot_bgcolor='rgb(240,240,240)',
                                                        margin=dict(t=150, b=150, l=50, r=50),
                                                        ))
                                           

app.layout = html.Div([
    html.H1('Hotel Embeddings Visualization', style={'text-align': 'center'}),
    html.Div([html.Label("Adjective-DF"), dcc.Dropdown(id='data-dropdown',
                               options=[{'label': label, 'value': label} for label in data],
                               value=list(dict_main.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '22%', 'display': 'inline-block'}),
        
    html.Div([html.Label("Method"), dcc.Dropdown(id='method-dropdown',
                               options=[{'label': label, 'value': label} for label in methods],
                               value=list(dict_method.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '22%', 'display': 'inline-block'}),
    html.Div([html.Label("Label-System"), dcc.Dropdown(id='label-dropdown',
                               options=[{'label': label, 'value': label} for label in labels],
                               value=list(dict_label.keys())[0],
                               multi=False,
                               searchable=False)], style={'width': '22%', 'display': 'inline-block'}),
        
    html.Div([html.Label("X-Component"), dcc.Dropdown(id='x-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_x],
                               value=list(dict_comp_x.keys())[0],
                               multi=False,
                               searchable=False)],style={'width': '11%', 'display': 'inline-block'}),
                                 
    html.Div([html.Label("Y-Component"), dcc.Dropdown(id='y-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_y],
                               value=list(dict_comp_y.keys())[0],
                               multi=False,
                               searchable=False)],style={'width': '11%', 'display': 'inline-block'}),
    html.Div([html.Label("Z-Component"), dcc.Dropdown(id='z-axis-dropdown', 
                               options=[{'label': label, 'value': label} for label in comp_z],
                               value=list(dict_comp_z.keys())[0],
                               multi=False,
                               searchable=False)],style={'width': '11%', 'display': 'inline-block'}),
    html.Div(html.A(id='download-link', children='Download File')),
    html.Center([
    html.Div(dcc_graph, style={'width': '48%', 'display': 'inline-block'}),
    html.Div(dcc_graph_2d, style={'width': '48%', 'display': 'inline-block'})])

        ])

@app.callback(
    [Output('download-link', 'children'),
     Output('Main-Graph', 'figure'),
     Output('Main-Graph-2d', 'figure')],
    [Input('data-dropdown', 'value'),
     Input('method-dropdown', 'value'),
     Input('label-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('z-axis-dropdown', 'value')],
    [State('Main-Graph', 'figure'),
     State('Main-Graph-2d', 'figure')])
def updateGraph(df_name, method_name, label_name, x_field, y_field, z_field, data, data_2d):
    source = data['data']
    source_2d = data_2d['data']
    df = dict_main[df_name]
    source[0].update({'customdata': df[tooltip_columns].values.tolist()})
    source_2d[0].update({'customdata': df[tooltip_columns].values.tolist()})
    
    with open("data.txt", "w") as text_file:
        text_file.write(str(data['data'][0]))
    with open("layout.txt", "w") as text_file:
        text_file.write(str(data['layout']))
        
    if x_field and y_field and z_field:
        source[0].update({'x': df[method_name+'_'+x_field].tolist(),
                          'y': df[method_name+'_'+y_field].tolist(),
                          'z': df[method_name+'_'+z_field].tolist()})
        source_2d[0].update({'x': df[method_name+'_'+x_field].tolist(),
                          'y': df[method_name+'_'+y_field].tolist()})
        source[0].update({'marker':{'color': df[label_name].tolist(),
                                    'size':4, 'colorscale':clr_scale}})
        source_2d[0].update({'marker':{'color': df[label_name].tolist(),
                                    'size':8, 'colorscale':clr_scale}})
        
        data['layout'].update({'title': {'text': str(method_name)+" | "+x_field+' vs '+y_field+' vs '+z_field}})
        data['layout']['xaxis'] = {**data['layout']['scene']['xaxis'], 'title': {'text': x_field}}
        data['layout']['yaxis'] = {**data['layout']['scene']['yaxis'], 'title': {'text': y_field}}
        data['layout']['zaxis'] = {**data['layout']['scene']['zaxis'], 'title': {'text': z_field}}
        data_2d['layout'].update({'title': {'text': str(method_name)+" | "+x_field+' vs '+y_field}})
        data_2d['layout']['xaxis'] = {**data_2d['layout']['xaxis'], 'title': {'text': x_field}}
        data_2d['layout']['yaxis'] = {**data_2d['layout']['yaxis'], 'title': {'text': y_field}}

    current_df = dict_main[df_name]
    current_df_name = "Hotel_" + df_name
    current_df.to_excel("./download/"+current_df_name+".xlsx", encoding='utf-8')
    url_string = './download/' +current_df_name+ '.xlsx'
    return [[html.A("Download link", href=url_string, target="_blank", download=url_string)],
            {'data': source, 'layout': data['layout']},
            {'data': source_2d, 'layout': data_2d['layout']}]

@app.server.route("/download/<path:path>")
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'download'), path)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, host="0.0.0.0", port=80)
