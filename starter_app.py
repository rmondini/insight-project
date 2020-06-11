# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import pickle

####################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

####################################

df_hotels = pd.read_csv('./datasets/input_app/df_hotels_unique_names.csv')
reg = pickle.load(open('price_regression_model.sav','rb'))
categorical_features = pickle.load(open('categorical_features.sav','rb'))
features_complete_list = pickle.load(open('features_complete_list.sav','rb'))
features_complete_list_dummies = pickle.load(open('features_complete_list_dummies.sav','rb'))

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
    df_hotel_item_cp[[str(n)+'_pc_neg' for n in range(-1,8)]] = 100*df_hotel_item_cp[[str(n)+'_neg' for n in range(-1,8)]].div(df_hotel_item_cp.sentences_count_neg, axis=0)
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
    #print('Price prediction :',str(round(price_pred,1)))
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
    html.Div(children='Choose one of the following hotels in New York:'),
    html.Div(children='[format: name (zip code)]'),   
    html.Br(),

    dcc.Dropdown(
        id='hotel-name',
        options=[{'label': i, 'value': i} for i in sorted(df_hotels['hotel_unique_name'].unique())],
        placeholder='Select a hotel...'
    ),
    html.Br(),

    html.Div('Negative mentions per topic:'),
    html.Div('-----------------------------------'),
    html.Div(id='hotel-neg-4'),
    html.Div(id='hotel-neg-3'),
    html.Div(id='hotel-neg-1'),
    html.Div(id='hotel-neg-2'),
    html.Div(id='hotel-neg-0'),
    html.Div(id='hotel-neg-6'),
    html.Div(id='hotel-neg-5'),
    html.Div(id='hotel-neg-7'),
    html.Div('-----------------------------------'),

    html.Br(),
    html.Div(children='Adjust the number of negative mentions per topic:'),
    html.Div(children='[0 (default): no change]'),
    html.Div(children='[-1: decrease number by 1]'),
    html.Div(children='[+1: increase number by 1]'),

    html.Br(),       
    html.Label('Location: '),
    dcc.Input(id='input-neg-4', value='0', type='text'),
    html.Label(' (e.g. surroundings, view, distance from attractions, transportation options)'),
    html.Br(),
    html.Label('Facilities: '),
    dcc.Input(id='input-neg-3', value='0', type='text'),
    html.Label(' (e.g. elevators, gym, pool, bar, restaurant, parking, wi-fi)'),
    html.Br(),
    html.Label('Staff: '),
    dcc.Input(id='input-neg-1', value='0', type='text'),
    html.Br(),
    html.Label('Breakfast: '),
    dcc.Input(id='input-neg-2', value='0', type='text'),
    html.Br(),
    html.Label('Room Comfort: '),
    dcc.Input(id='input-neg-0', value='0', type='text'),
    html.Label(' (e.g. noise, ac/heating, smell)'),
    html.Br(),
    html.Label('Room Amenities: '),
    dcc.Input(id='input-neg-6', value='0', type='text'),
    html.Label(' (e.g. tv, fridge, coffee/tea maker, room service, appearance, furniture)'),
    html.Br(),
    html.Label('Bathroom: '),
    dcc.Input(id='input-neg-5', value='0', type='text'),
    html.Br(),
    html.Label('Bed Quality: '),
    dcc.Input(id='input-neg-7', value='0', type='text'),
    html.Br(),html.Br(),

    html.Div(id='price-prediction')


])

####################################

# @app.callback(
#     Output('neg-mentions-header','children'),
#     [Input('hotel-name', 'value')])
# def print_neg_mention_header(hotel_name):
#     return 'Negative mentionssss per topic:'

####################################

@app.callback(
    Output('hotel-neg-0','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_0(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Room Comfort: ' + str(int(hotel_item['0_neg']))

@app.callback(
    Output('hotel-neg-1','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_1(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Staff: ' + str(int(hotel_item['1_neg']))

@app.callback(
    Output('hotel-neg-2','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_2(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Breakfast: ' + str(int(hotel_item['2_neg']))

@app.callback(
    Output('hotel-neg-3','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_3(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Facilities: ' + str(int(hotel_item['3_neg']))

@app.callback(
    Output('hotel-neg-4','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_4(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Location: ' + str(int(hotel_item['4_neg']))

@app.callback(
    Output('hotel-neg-5','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_5(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Bathroom: ' + str(int(hotel_item['5_neg']))

@app.callback(
    Output('hotel-neg-6','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_6(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Room Amenities: ' + str(int(hotel_item['6_neg']))

@app.callback(
    Output('hotel-neg-7','children'),
    [Input('hotel-name', 'value')])
def print_hotel_neg_7(hotel_name):
    hotel_item = df_hotels[df_hotels['hotel_unique_name']==hotel_name]
    return 'Bed Quality: ' + str(int(hotel_item['7_neg']))

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
        return 'The model predicts that the average room price can be increased by {}%'.format(round(price_output[2],1))
    elif price_output[2]<0.0:
        return 'The model predicts that the average room price should be decreased by {}%'.format(round(abs(price_output[2]),1))  
    else:
        return 'The model predicts no change in the average room price' 

####################################

if __name__ == '__main__':
    app.run_server(debug=True)