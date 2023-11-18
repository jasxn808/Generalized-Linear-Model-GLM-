import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import altair as alt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from plotly import express as px



st\
    .title('3rd Down Conversion Rates over Expected Analysis')

st\
    .subheader('Introduction:')

st\
    .write('This interactive Streamlit App allows users to see the Conversion Rates over Expected (CROE) on 3rd downs for NFL quarterbacks.' )
# st\
#     .write('\n')
st\
    .write('Using Machine Learning, we are then able to predict new Conversion Rates given actual game-time scenarios by adding additional features into our model.')

st\
    .write('This app begins with using "Yards to Go" as the only feature, and builds from there based on user inputs.')

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


#Data Preview - conversion rate based on ydstogo:
st.subheader('Data Preview - Conversion Rates based on Yards to Go')

st\
    .write("To simulate most common game scenarios, let's set the maximum distance at 3rd-and-15 and include both completed/incompleted passes.")
st\
    .write("Here's a preview of our dataset:")

#play features: pass play on 3rd down, pass = complete & incomplete, air_yards = tracked
pbp_py_third = \
    pbp_py\
        .query('play_type == "pass" & down == 3.0 & air_yards.notnull() & ydstogo <= 15')

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
    .subheader('Generalized Linear Model (GLM) using "Yards to Go" as only feature:')

st\
    .write("Using ""Yards to Go"" as our only feature (for now), let's use a Generalized Linear Model (GLM) to predict Third-Down Conversion Rates.")

st\
    .write("A pass on 3rd down only has 2 outcomes - either Completed or Incompleted. Likewise, a conversion on 3rd down only has 2 outcomes - either Converted or Failed.")

st\
    .write('At a high level, to predict the probability of a statistic bounded by 0 - 1, we will be using a GLM with Logistic Regression to create our predictions.')

conversion_py = \
  smf.glm(formula='third_down_converted ~ ydstogo',
          data=pbp_py_third,
          family=sm.families.Binomial())\
          .fit()

st\
    .write(conversion_py.summary())

st\
    .subheader('Conversion Rate over Expected (CROE) Explained:')
st.write('From our GLM model, we now have a predicted Third Down Conversion Rate based on Yards to Go as our feature.')
st.write('To calculate CROE, we will take the actual Third Down Conversion Outcome (1 or 0) and subtract our predicted rate from our model. Afterwards, we will aggregate to get a mean CROE value for the season, per QB.')
st.write("Let's quickly see our new dataset and see some data visualization for starting NFL QBs (at least 50 converted 3rd Downs as a benchmark):")
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
       'third_down_converted':['mean','count']})


croe_leaders.columns = \
  list(map('_'.join, croe_leaders.columns))

croe_leaders.reset_index(inplace=True)
croe_leaders.rename\
  (columns = {'CROE_mean':'CROE_avg', 'third_down_converted_mean':'avg_conv_rate', 'third_down_converted_count':'n'}, inplace=True)


top10_croe = \
    croe_leaders.query('n>50').sort_values(by='CROE_avg', ascending=False)

top10_croe_display = \
    top10_croe.rename(columns = {'CROE_avg':'Avg. CROE','avg_conv_rate':'Avg. Conversion', 'n':'Count Conversions'})

st.write(top10_croe_display)

##graphing:

plt.style.use('ggplot')


x_ax=top10_croe['passer_player_name']
y_ax=top10_croe['CROE_avg']

col = [{x<0:'red', x>0:'green'}[True] for x in y_ax]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

plt.xlabel('Mean Conversion Rate over Expected (CROE)')
plt.title(f'QBs with Best/Worst CROE in {season_var}')


plt.tight_layout()
ax.barh(x_ax,y_ax, color=col)
st.write(fig)



#Adding more features to model:
st\
    .subheader('Adding Additional Features to our GLM model:')

st\
    .write('For some interactivity, the user can now include additional, game-realistic features that influence Third Down Conversion Rates:')

if 'features_list' not in st.session_state:
    st.session_state.features_list = \
        ['Yards to Go']

# st\
#     .write('My current features to include are:', st.session_state.features_list)

add_features = \
    st.selectbox('What other features would you like to include?:',\
                 ['QB Hit?', 'Number of Pass Rushers', 'Yards to go for TD', 'Air Yards Thrown'])
if st.button('Add new feature to GLM Model'):
    st.session_state.features_list.append(add_features)
if st.button('Remove your last choice:'):
    st.session_state.features_list.pop()
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
    elif var == 'Air Yards Thrown':
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


st.write(conversion_v2_py.summary())
#calculating CROE after adding addt'l features:
pbp_py_third_compl['exp_third_conv'] = \
  conversion_v2_py.predict()

pbp_py_third_compl['CROE'] =\
  pbp_py_third_compl['third_down_converted'] - pbp_py_third_compl['exp_third_conv']


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
    .subheader('3rd Down Conversion Rate over Expected Leaders')
st\
    .write('From our new model calculations, here are the best/worst QB CROE performances - again using > 50 successful Third Down Conversions as our benchmark.')

qb1 = \
    pbp_py_third_convr.sort_values(by='CROE', ascending=False).query('n > 50')


qb1_display = \
    qb1.rename(columns = {'avg_conversion':'Avg. Conversion', 'n':'Count Conversions', 'CROE':'Avg. CROE'})
st\
    .write(qb1_display)





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

col2 = [{x<0:'red', x>0:'green'}[True] for x in y_ax]
plt.tight_layout()
ax.barh(x_ax,y_ax,color=col2)
st.write(fig)


# c = (
#     alt.Chart(qb1)
#     .mark_circle()
#     .encode(x='passer_player_name',y='CROE')
# )
# st.altair_chart(c, use_container_width=True)



pos_CROE = px.scatter(qb1.query('CROE > 0'), x="avg_conversion", y="CROE",
	         size="CROE", color="passer_player_name",
                 hover_name="passer_player_name", log_x=True, size_max=60,
                 labels = {
                     'avg_conversion':'Mean Conversion Rate',
                     'CROE' : 'Mean CROE Rate',
                     'passer_player_name':'QB'
                 },
                 title = 'Mean Conversion Rate vs. CROE (Positive Values)')

st.write(pos_CROE)


st.subheader('Conclusion')
st\
    .write("Completion Rate over Expected should not be the end-all, be-all metric for QB performance categorization. For example, in 2022, Tom Brady and Aaron Rodgers both had lower CROEs than Kenny Pickett."
        " Does that mean Kenny Pickett is the better QB? Most likely not.")
st\
    .write('\n')

st\
    .write("Looking at the co-efficients produced by our second GLM model, it seems the features we have included as options may not be the most statistically significant features to include."
           "In the future, this model could be further improved by identifying the most significant features that affect 3rd Down Conversion rates.")

st\
    .write('\n')

st\
    .write("3rd Down conversions are influenced by other factors beyond the scope of the GLM models"
           "- coaching, supporting cast, weather, etc. Instead, CROE should be used as a metric tool as part of a larger evaluation process in determining QB performance.")


