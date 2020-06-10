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

def display_info_hotel(df_hotels_item):
    
    return '----------------' + '\n' + 'Negative mentions per topic: '

    # print('Hotel name: ',df_hotels_item['hotel_name'].values[0])
    # print('---------------------')
    # print('Negative mentions per topic: ')
    # print('(0)  Noise/Smell/AC-Heat: ', str(int(df_hotels_item['0_neg'])))
    # print('(1)  Staff/Check-in/out: ', str(int(df_hotels_item['1_neg'])))
    # print('(2)  Breakfast: ', str(int(df_hotels_item['2_neg'])))
    # print('(3)  Facilities: ', str(int(df_hotels_item['3_neg'])))
    # #print('(4)  Parking: ', str(int(df_hotels_item['4_neg'])))
    # #print('(5)  Smell: ', str(int(df_hotels_item['5_neg'])))
    # #print('(6)  AC/Heat: ', str(int(df_hotels_item['6_neg'])))
    # #print('(7)  Wi-Fi: ', str(int(df_hotels_item['7_neg'])))
    # print('(4)  Location: ', str(int(df_hotels_item['4_neg'])))
    # #print('(9)  Check-in/Check-out: ', str(int(df_hotels_item['9_neg'])))
    # print('(5) Bathroom: ', str(int(df_hotels_item['5_neg'])))
    # print('(6) Room amenities: ', str(int(df_hotels_item['6_neg'])))
    # print('(7) Bed: ', str(int(df_hotels_item['7_neg'])))
    # #print('Other/no topic: ', str(int(df_hotels_item['-1_neg'])))

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
    #df_hotel_item_cp.at[0,'8_neg'] += counts_delta_vec[8]
    #df_hotel_item_cp.at[0,'9_neg'] += counts_delta_vec[9]
    #df_hotel_item_cp.at[0,'10_neg'] += counts_delta_vec[10]
    #df_hotel_item_cp.at[0,'11_neg'] += counts_delta_vec[11]
    #df_hotel_item_cp.at[0,'12_neg'] += counts_delta_vec[12]
    #df_hotel_item_cp.at[0,'sentences_count_neg'] +=sum(counts_delta_vec)
    df_hotel_item_cp[[str(n)+'_pc_neg' for n in range(-1,8)]] = 100*df_hotel_item_cp[[str(n)+'_neg' for n in range(-1,8)]].div(df_hotel_item_cp.sentences_count_neg, axis=0)
    
    #display_info_hotel(df_hotel_item_cp)
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
        style={
            'textAlign': 'center'
        }
    ),

    html.Br(),
    html.Div(children='Choose one of the following hotels in New York:'),
    html.Div(children='[format: name (zip code)]'),   

    dcc.Dropdown(
        id='hotel-name',
        options=[{'label': i, 'value': i} for i in sorted(df_hotels['hotel_unique_name'].unique())],
        placeholder='Select a hotel...'
    ),
    html.Br(),

    html.Div(id='hotel-name-output'),

    html.Label('Room Comfort: '),
    dcc.Input(id='input-neg-0', value='0', type='text'),
    html.Br(),
    html.Label('Staff: '),
    dcc.Input(id='input-neg-1', value='0', type='text'),
    html.Br(),
    html.Label('Breakfast: '),
    dcc.Input(id='input-neg-2', value='0', type='text'),
    html.Br(),
    html.Label('Facilities: '),
    dcc.Input(id='input-neg-3', value='0', type='text'),
    html.Br(),
    html.Label('Location: '),
    dcc.Input(id='input-neg-4', value='0', type='text'),
    html.Br(),
    html.Label('Bathroom: '),
    dcc.Input(id='input-neg-5', value='0', type='text'),
    html.Br(),
    html.Label('Room Amenities: '),
    dcc.Input(id='input-neg-6', value='0', type='text'),
    html.Br(),
    html.Label('Bed: '),
    dcc.Input(id='input-neg-7', value='0', type='text'),
    html.Br(),html.Br(),

    html.Div(id='price-prediction')


])

@app.callback(
    Output('hotel-name-output','children'),
    [Input('hotel-name', 'value')])
def print_hotel_info(hotel_name):

    #display_info_hotel(df_hotels[df_hotels['hotel_unique_name']==hotel_name])

    return hotel_name

@app.callback(
    Output('price-prediction','children'),
    [Input('hotel-name-output', 'children'),
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