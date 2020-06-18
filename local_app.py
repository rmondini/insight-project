
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import pickle
import plotly.graph_objects as go

####################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

####################################

df_hotels = pd.read_csv('./input_app/df_hotels_unique_names.csv')
reg = pickle.load(open('./input_app/price_regression_model.sav','rb'))
categorical_features = pickle.load(open('./input_app/categorical_features.sav','rb'))
features_complete_list = pickle.load(open('./input_app/features_complete_list.sav','rb'))
features_complete_list_dummies = pickle.load(open('./input_app/features_complete_list_dummies.sav','rb'))

####################################

def change_topic_counts(df_hotel_item,counts_delta_vec):

    df_hotel_item_cp = df_hotel_item.copy()
    df_hotel_item_cp.reset_index(drop=True,inplace=True)
    df_hotel_item_cp.at[0,'0_neg'] += counts_delta_vec[0]
    df_hotel_item_cp.at[0,'1_neg'] += counts_delta_vec[1]
    df_hotel_item_cp.at[0,'2_neg'] += counts_delta_vec[2]
    df_hotel_item_cp.at[0,'3_neg'] += counts_delta_vec[3]
    df_hotel_item_cp.at[0,'4_neg'] += counts_delta_vec[4]
    df_hotel_item_cp.at[0,'5_neg'] += counts_delta_vec[5]
    df_hotel_item_cp.at[0,'6_neg'] += counts_delta_vec[6]
    df_hotel_item_cp.at[0,'7_neg'] += counts_delta_vec[7]
    df_hotel_item_cp[[str(n)+'_pc_neg' for n in range(-1,8)]] = 100*df_hotel_item_cp[[str(n)+'_neg' for n in range(-1,8)]].div(df_hotel_item_cp.sentences_count_neg,axis=0)

    return df_hotel_item_cp

def get_price_prediction(model,cat_feat,feat_compl_list,feat_compl_list_dum,df_hotel_item):
    
    input_features = df_hotel_item[feat_compl_list]
    input_target = df_hotel_item[['hotel_room_price_per_person_avg']]

    scaled_input = input_features.copy()
    scaled_input = scaled_input.reindex(columns = feat_compl_list_dum, fill_value=0)
    for cat in cat_feat:
        dummy_cat = cat+'_'+str(input_features[cat].values[0])
        if dummy_cat in scaled_input.columns:
            scaled_input.iloc[0, scaled_input.columns.get_loc(dummy_cat)] = 1
            
    price_pred = model.predict(scaled_input)[0][0]
    return price_pred

def update_price(model,cat_feat,feat_compl_list,feat_compl_list_dum,df_hotel_item,counts_delta_vec):

    # baseline price from model
    baseline_price = get_price_prediction(model,cat_feat,feat_compl_list,feat_compl_list_dum,df_hotel_item)
    # change negative mentions per topic
    df_hotel_item_changed = change_topic_counts(df_hotel_item,counts_delta_vec)
    # new price from model
    new_price = get_price_prediction(model,cat_feat,feat_compl_list,feat_compl_list_dum,df_hotel_item_changed)
    # per cent change
    per_cent_change = 100*(new_price-baseline_price)/baseline_price
    
    return [baseline_price,new_price,per_cent_change]

####################################

app.layout = html.Div([

    html.H1(
        children='Hotel Review Analyzer',
        style={'textAlign': 'center'}
    ),

    html.Div(children='an interactive tool that helps hotel',
             style={'textAlign': 'center'}),
    html.Div(children='managers answer the question:',
             style={'textAlign': 'center'}),
    html.Div(children='"how much do bad reviews cost?"',
             style={'textAlign': 'center'}),

    html.Br(),html.Br(),html.Br(),

    dcc.Markdown('''
Choose one of the following hotels in New York:  
\[format: name (zip code)\]
    '''),

    dcc.Dropdown(
        id='hotel-name',
        options=[{'label': i, 'value': i} for i in sorted(df_hotels['hotel_unique_name'].unique())],
        placeholder='Select a hotel...'
    ),

    html.Div(id='hotel-topic-table'),

    dcc.Markdown('''
Adjust the numbers for each topic:
- 0 (default): no change
- -1: decrease number by 1
- +1: increase number by 1
    '''),

    html.Br(),       
    html.Label('Location: '),
    dcc.Input(id='input-neg-4', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Facilities: '),
    dcc.Input(id='input-neg-3', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Staff: '),
    dcc.Input(id='input-neg-1', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Breakfast: '),
    dcc.Input(id='input-neg-2', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Room Comfort: '),
    dcc.Input(id='input-neg-0', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Room Amenities: '),
    dcc.Input(id='input-neg-6', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Bathroom: '),
    dcc.Input(id='input-neg-5', value='0', type='text', style={'width':40}),
    html.Br(),
    html.Label('Bed Quality: '),
    dcc.Input(id='input-neg-7', value='0', type='text', style={'width':40}),
    html.Br(),html.Br(),

    html.Div(id='price-prediction'),
    html.Br(),html.Br()

])

####################################

@app.callback(
    Output('hotel-topic-table','children'),
    [Input('hotel-name', 'value')])
def print_hotel_topic_table(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]

    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Topic</b>', '<b>Negative mentions</b>'],fill_color='lightgray'),
                                   cells=dict(values=[['Location','Facilities','Staff','Breakfast','Room Comfort',
                                   	                   'Room Amenities','Bathroom','Bed Quality'],
                                   	                  [int(hotel_item['4_neg']),int(hotel_item['3_neg']),int(hotel_item['1_neg']),int(hotel_item['2_neg']),
                                                       int(hotel_item['0_neg']),int(hotel_item['6_neg']),int(hotel_item['5_neg']),int(hotel_item['7_neg'])]]))
                    ])
    fig.update_layout(width=300,
    	              height=220,
    	              margin=dict(l=1,r=1,t=20,b=1))

    return html.Div([dcc.Graph(id='topic-table',figure=fig)])

####################################

@app.callback(
    Output('price-prediction','children'),
    [Input('hotel-name', 'value'),
     Input('input-neg-0', 'value'),
     Input('input-neg-1', 'value'),
     Input('input-neg-2', 'value'),
     Input('input-neg-3', 'value'),
     Input('input-neg-4', 'value'),
     Input('input-neg-5', 'value'),
     Input('input-neg-6', 'value'),
     Input('input-neg-7', 'value')])
def update_price_wrapper(hotel_name,input_neg_0,input_neg_1,input_neg_2,input_neg_3,input_neg_4,input_neg_5,input_neg_6,input_neg_7):

    input_hotel = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    neg_0_int = int(input_neg_0)
    neg_1_int = int(input_neg_1)
    neg_2_int = int(input_neg_2)
    neg_3_int = int(input_neg_3)
    neg_4_int = int(input_neg_4)
    neg_5_int = int(input_neg_5)
    neg_6_int = int(input_neg_6)
    neg_7_int = int(input_neg_7)

    if (int(input_hotel['0_neg'])+neg_0_int<0) or (int(input_hotel['1_neg'])+neg_1_int<0) or (int(input_hotel['2_neg'])+neg_2_int<0) or \
       (int(input_hotel['3_neg'])+neg_3_int<0) or (int(input_hotel['4_neg'])+neg_4_int<0) or (int(input_hotel['5_neg'])+neg_5_int<0) or \
       (int(input_hotel['6_neg'])+neg_6_int<0) or (int(input_hotel['7_neg'])+neg_7_int<0):
        return 'You cannot have a negative number of mentions for a topic'

    counts_delta_vec = [neg_0_int,neg_1_int,neg_2_int,neg_3_int,neg_4_int,neg_5_int,neg_6_int,neg_7_int]

    price_output = update_price(reg,categorical_features,features_complete_list,features_complete_list_dummies,input_hotel,counts_delta_vec)

    if price_output[2]>0.0:
        return html.Div(children=dcc.Markdown('The model predicts that room prices can be **increased by ' + str(round(price_output[2],1)) + '%**'))
    elif price_output[2]<0.0:
        return html.Div(children=dcc.Markdown('The model predicts that room prices should be **decreased by ' + str(round(abs(price_output[2]),1)) + '%**')) 
    else:
        return html.Div(children=dcc.Markdown('The model predicts **no** change in room prices'))

####################################

if __name__ == '__main__':
    app.run_server(debug=True)