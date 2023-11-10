import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import altair as alt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

st\
    .title('3rd Down Conversion Rates over Expected Analysis')

st\
    .subheader('Introduction:')

st\
    .write('This interactive Streamlit App allows users to see the Conversion Rates over Expected (CROE) for NFL quarterbacks' )
# st\
#     .write('\n')
st\
    .write('Using Machine Learning, we are able to then predict Conversion Rates given game-time scenarios by adding additional features into our model')

st\
    .write('This app begins with using "Yards to Go" as the only feature, and builds from there based on user inputs')

st\
    .write('**Created By: Jason Park**')

#user input: select nfl season to analyze:

st\
    .subheader('Season Selection:')

season_var =\
    st.selectbox('Select an NFL season to analyze:',\
                 [2016, 2017, 2018, 2019, 2020, 2021, 2022])


##caching??? nned to confirm if this is working as intended
#st.cache_data
def load_pbp():
    df = \
        nfl.import_pbp_data([season_var])
    return(df)


pbp_py = \
    load_pbp()

pbp_py_third = \
    pbp_py\
        .query('play_type == "pass" & down == 3.0 & air_yards.notnull() & ydstogo <= 15')


#Data Preview - conversion rate based on ydstogo:
st.subheader('Data Preview - Conversion Rates based on Yards to Go')

st\
    .write("To simulate most common game scenarios, let's set the maximum distance at 3rd-and-15 and include both completed/incompleted passes")

df_conv = \
pbp_py_third \
    .groupby('ydstogo')\
    .agg({'third_down_converted':['mean', 'count']})

df_conv.columns = list(map('_'.join, df_conv.columns))
df_conv.reset_index(inplace=True)

df_conv.rename(columns = \
    {'third_down_converted_mean':'conversion_rate', 'third_down_converted_count':'n'}, inplace=True)

df_conv_display = \
df_conv.rename(columns = \
                {'ydstogo':'Yards to Go'
                 ,'conversion_rate':'Average Conv. Rate'
                 ,'n':'Attempts'})

df_conv_display['Average Conv. Rate'] = ((df_conv_display['Average Conv. Rate']*100).round(2))

st.write(df_conv_display)

st.bar_chart(data=df_conv, x='ydstogo', y='conversion_rate')

# st\
# .write(plt.bar(df_conv_display['Yards to Go'], df_conv_display['Average Conv. Rate']))


#plotting linear plot to show trend:
# st\
#     .subheader('Linear Trend Line Visualization:')
# fig_sb, ax_sb = plt.subplots()
# ax_sb = \
#  sns.regplot\
#   (data=df_conv
#    , x='ydstogo'
#    , y='conversion_rate'
#    , line_kws={'color':'blue'})
# ax_sb.set(xticks=np.arange(0,16,1))   

# st.pyplot(fig_sb)





##GLM:

st\
    .subheader('GLM Regression using "Yards to Go" as only feature:')

conversion_py = \
  smf.glm(formula='third_down_converted ~ ydstogo',
          data=pbp_py_third,
          family=sm.families.Binomial())\
          .fit()

st\
    .write(conversion_py.summary())

st\
    .subheader('Best/Worst Expected Conversion Rate using GLM')
#expected conversion rate based on ydstogo:

pbp_py_third['exp_third_conv'] = \
  conversion_py.predict()

#CROE = Conversion Rate over Expected
pbp_py_third['CROE'] = \
  pbp_py_third['third_down_converted'] - pbp_py_third['exp_third_conv']



croe_leaders = \
pbp_py_third\
  .groupby(['season', 'passer_player_id', 'passer_player_name'])\
  .agg({'CROE':['mean'],
       'third_down_converted':['mean','sum']})


croe_leaders.columns = \
  list(map('_'.join, croe_leaders.columns))

croe_leaders.reset_index(inplace=True)
croe_leaders.rename\
  (columns = {'CROE_mean':'CROE_avg', 'third_down_converted_mean':'avg_conv_rate', 'third_down_converted_sum':'n'}, inplace=True)

#greater than 50 third down attempts
st\
    .write(croe_leaders.sort_values(by='CROE_avg', ascending=False).query('n>50'))

top10_croe = \
    croe_leaders.query('n>50').sort_values(by='CROE_avg', ascending=False)

##graphing:

plt.style.use('fivethirtyeight')


x_ax=top10_croe['passer_player_name']
y_ax=top10_croe['CROE_avg']


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

plt.xlabel('Conversion Rate over Expected (CROE)')
plt.title(f'QBs with Best/Worst CROE in {season_var}')


plt.tight_layout()
ax.barh(x_ax,y_ax)
st.write(fig)



#Adding more features to model:
st\
    .subheader('Adding additional features to our GLM model:')

if 'features_list' not in st.session_state:
    st.session_state.features_list = \
        ['Yards to Go']

# st\
#     .write('My current features to include are:', st.session_state.features_list)

add_features = \
    st.selectbox('What other features would you like to include?:',\
                 ['QB Hit?', 'Number of Pass Rushers', 'Yards to go for TD', 'Air Yards (for QB throws)'])
if st.button('Add new feature to GLM Model'):
    st.write('Adding new feature to features list')
    st.session_state.features_list.append(add_features)
st.write('New features to be included are: ', \
            st.session_state.features_list)

glm_features = ['passer', 'passer_id', 'passer_player_name'\
                       ,'season', 'down', 'third_down_converted']
glm_features_string = []
for var in st.session_state.features_list:
    if var == 'Yards to Go':
        var = 'ydstogo'
    elif var == 'QB Hit?':
        var = 'qb_hit'
    elif var == 'Number of Pass Rushers':
        var = 'number_of_pass_rushers'
    elif var == 'Yards to go for TD':
        var = 'yardline_100'
    elif var == 'Air Yards (for QB throws)':
        var = 'air_yards'
    glm_features.append(var)
    glm_features_string.append(var)



# st.write(pbp_py_third[glm_features].head())
pbp_py_third_compl = \
    pbp_py_third[glm_features].dropna(axis=0)

##GLM summary with additional features:
st\
    .subheader('GLM Model using additional features added:')

formula_str = 'third_down_converted~'
for s in glm_features_string:
    formula_str += s + '+'
formula_str = formula_str[:-1]


conversion_v2_py = \
  smf.glm(formula=formula_str
          , data=pbp_py_third_compl
          , family=sm.families.Binomial())\
          .fit()
st\
    .write(conversion_v2_py.summary())


#calculating CROE after adding addt'l features:
pbp_py_third_compl['exp_third_conv'] = \
  conversion_v2_py.predict()

pbp_py_third_compl['CROE'] =\
  pbp_py_third['third_down_converted'] - pbp_py_third['exp_third_conv']


pbp_py_third_convr = \
pbp_py_third_compl\
  .groupby(['passer_id', 'passer_player_name', 'season'])\
  .agg({'CROE':['mean'],\
        'third_down_converted':['mean','count']})

pbp_py_third_convr.columns = list(map('_'.join, pbp_py_third_convr))
pbp_py_third_convr.reset_index(inplace=True)

pbp_py_third_convr\
  .rename(columns={'CROE_mean':"CROE", 'third_down_converted_mean':'avg_conversion','third_down_converted_count':'n'}\
          ,inplace=True)

st\
    .subheader('3rd Down Conversion Rate over Expected Leaders (over 100 passes)')
qb1 = \
    pbp_py_third_convr.sort_values(by='CROE', ascending=False).query('n > 100')
st\
    .write(qb1)


st\
    .subheader('Data Visualization:')


#Bar Chart Matplotlib
plt.style.use('ggplot')

x_ax=qb1['passer_player_name']
y_ax=qb1['CROE']


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

plt.xlabel('Conversion Rate over Expected (CROE)')
plt.title(f'QBs with Best/Worst CROE in {season_var}')


plt.tight_layout()
ax.barh(x_ax,y_ax)
st.write(fig)



