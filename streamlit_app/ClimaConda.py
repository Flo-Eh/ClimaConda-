#Package importation 

import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import warnings
import statsmodels.api as sm
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#Page configuration 
st.set_page_config(page_title="ClimaConda",page_icon='🌍',layout="wide", initial_sidebar_state="auto")

#Barre de Navigation 
st.sidebar.title('Navigation')
#Différentes pages du site 
pages = ['Introduction','Data exploration',' Models','Try it yourself !!!' , 'Conclusion']
page = st.sidebar.radio(' ',pages)

#Importation du df sur l'europe 
df = pd.read_csv('assets/final_df_UE.csv',index_col = 'year')

#df.drop(['Unit','Unnamed: 0'], axis = 1, inplace = True)


                                                     #Page 1: Introduction 
    
    
    
if page == pages[0]:
#Project Title
    st.markdown("<h1 style='text-align: center; color: green;'>ClimaConda</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: blue;'>DataScientest Project / Bootcamp October 2022</h2>", unsafe_allow_html=True)
 #Team 
    st.subheader('By Florian Ehre, Michael Laidet & Mai Anh Vu')
 #Picture   
    st.image('assets/rechauffement.png', width = 1200)
 #Project description    
    st.write(
        """
        Based on a research paper published in 2022, this project aims to link environmental issues with Machine Learning by predicting greenhouse gas emissions for the next 10 years. \n
        
        The research paper in question tries to find the best possible model to evaluate these emissions in India, by testing different models:
        - Statistical (ARIMA, SARIMAX, HOLT-WINTER),
        - Machine Learning (Linear Regression, Random Forest),
        - Deep Learning (LSTM)
        
        We followed a similar approach, focusing first on a specific gas and sector : CO2 emissions of the Transportion sector in France. \n
        Then we industrialized our process so that any emissions series can go through all our models in one click. \n
        In the end, we could easily compare our global emissions forecasts with the EU's greenhouse gas emission reduction target (-40% by 2030 compared to 1990) to measure the effort to be made.

        """
    )
    
    
    
    
    
                                                     #Page 2: Data Viz    
if page == pages[1]:  
#Title
    st.title('Data exploration') 

#Dataset
    st.header('Dataset')

    st.write(
        """
        We used the European Union emissions data from **1990 to 2019** published by [Climate Watch](https://www.climatewatchdata.org/ghg-emissions). \n
        Emissions are expressed in Mt carbon equivalent and are calculated as the quantities of GHGs physically emitted within the country (territorial approach) by households (cars and housing) and economic activities (fossil energy consumption, industrial processes and agricultural emissions).\n
        
        Data can be analyzed at different levels:
        - by country
        - by sector: Agriculture, Building, Bunker Fuels, Electricity/Heat, Energy, Fugitive Emissions, Industrial Processes, Land-Use Change and Forestry, Manufacturing/Construction, Other Fuel Combustion, Transportation, Waste
        - by gas: Carbon dioxide (CO2), Methane (CH4), Nitrous oxide(N2O), Fluorinated gases (F-Gas)
        """    
    )

#Preprocessing
    st.header('Preprocessing')

    st.write(
        """
        To use Climate Watch data for visualization and models, we brought a few transformations : years were transposed to lines, defined as index and converted in datetime format.
        Useless columns were removed.
    
        """)
    
    df_viz = df.sort_values(['year','Country','Sector', 'Gas'])
    st.dataframe(df_viz.head(10))

#Dataviz
    st.header('Data visualization')

    st.write(
        """
        For visualization, we chose to analyze France emissions data.
        """)    

    df_france = df[df['Country']== 'France']
    sectors = ['Agriculture', 'Building', 'Bunker Fuels' ,'Electricity/Heat','Fugitive Emissions', 'Industrial Processes',
    'Manufacturing/Construction','Other Fuel Combustion','Transportation', 'Waste']

    sectors_2019 = df_france[(df_france['Sector'].isin(sectors)) &
                          (df_france.index == 2019) & 
                          (df_france['Gas'] == 'All GHG')]

    gas_2019 = df_france[(df_france.index == 2019) & 
                     (df_france['Sector'] == 'Total excluding LUCF')&
                     (df_france['Gas']!= 'All GHG')].groupby('Gas').sum()                     
    
    fig1 = plt.figure(figsize = (12,4))

    plt.subplot(121)
    plt.pie(sectors_2019['cons'], 
        labels = sectors_2019['Sector'],
        autopct = lambda x:str(round(x,2)) +'%',
        pctdistance = .7)
    plt.title('Distribution of 2019 GHG emissions per sector in France')

    plt.subplot(122)
    plt.pie(gas_2019['cons'], 
        labels = gas_2019.index,
        autopct = lambda x:str(round(x,2)) +'%',
        pctdistance = .8)
    plt.title('Distribution of 2019 GHG emissions per gas in France')

    with st.container():
        st.markdown("<h4>Distribution of global emissions per sector and per gas in France in 2019</h4>", unsafe_allow_html=True)
        st.pyplot(fig1)

        st.info(
            """
        → C02 is by far the most important greenhouse gas emitted in France in 2019\n
        → Transportation sector is the biggest emitter of greenhouse gasses in France, followed by Agriculture and Electricity/Heat sectors_
        """)

    # Evolution of France global emissions per sector (including LUCF in absolute value)

    global_sectors = df_france[(df_france['Gas']=='All GHG') &
                   (df_france['Sector'].isin(sectors))]

    global_sectors= global_sectors.reset_index()
    
    lucf = df_france[(df_france['Gas']=='All GHG') &
                   (df_france['Sector']== 'Land-Use Change and Forestry')]

    lucf['cons'] = np.abs(lucf['cons'])
    lucf= lucf.reset_index()
    
    fig2, ax = plt.subplots(1,1, figsize = (12,6))
    sns.lineplot(x = 'year', y = 'cons',data= global_sectors, hue = 'Sector', ax = ax)
    sns.lineplot(x= 'year', y = 'cons',data = lucf, linestyle = '--', linewidth = 2, color = 'black', label = 'Land-Use Change and Forestry', ax = ax)                    
    ax.fill_between(lucf['year'], lucf['cons'], color='k', alpha=0.05)

    plt.title('Global GHG emissions evolution per sector (1990-2019)')
    plt.ylabel('Emissions in MtCO₂e')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1));
        
    st.markdown("<h4>Evolution of global emissions per sector in France since 1990</h4>", unsafe_allow_html=True)
    st.pyplot(fig2)

    st.info(
        """
    LUCF emissions were negative over the whole period, meaning that the sector is absorbing more GHG than emitting.
    To illustrate the compensating effect of LUCF we used the LUCF emissions absolute value in the graph.

    → Building, Electricity/Heat and Manufacturing/Constructions sectors have reduced their emissions since 2000 \n
    → On the contrary, the emissions of the Transportation sector - the biggest emitter -  continue to slightly increase \n
    → Since 2001, LUCF compensates the emissions of Manufacturing/Construction
    """
    )

    all_gas= df_france[df_france['Gas']!= 'All GHG']
    all_gas = all_gas.reset_index()
    crosstab_gas = pd.crosstab(all_gas['year'], all_gas['Gas'], values = all_gas['cons'],aggfunc = 'sum')
    

    fig3, ax = plt.subplots(figsize=(12,6))
    crosstab_gas.plot.bar(y=['CH4','CO2', 'F-Gas', 'N2O'], stacked = True, ax = ax)
    plt.xticks(rotation = 50)
    plt.title('Distribution of global GHG emissions per gas (1990-2019)')
    plt.ylabel('Emissions in MtCO₂e')
    plt.xlabel('Year')
    plt.legend(loc='upper right')

    st.markdown("<h4>Evolution of the distribution of emissions per gas in France since 1990</h4>", unsafe_allow_html=True)
    st.pyplot(fig3)

    st.info(
        """
        → The significant decrease of C02 emissions since 2000 explains the decrease of global GHG emissions \n
        → The emissions of the other gasses are quite stable since 1990
    """
    )


                                                    #Page 3: Models    
if page == pages[2]:
    #Title
    st.title('Models') 

    st.header('Problematic')

    st.write(
    """
    We are working on a univariate time series and we want to forecast emissions for the next 10 years.

    To begin with, we focused on CO2 emissions of the transportation sector, which is the main contributor of GHG in France.
    """
    )
    
    st.image('assets/series.png')

    st.write(
    """
    To measure the performance of our models we chose MAE (Mean Absolute Error) which is an indicator of the accuracy of the predictions and is easy to interpret.
    We also computed MAPE (Mean Absolute Percentage Error), RMSE (Root Mean Squared Error) and r2 to compare the performance of our models against different metrics.

    We tested several approaches - statistical, machine learning, deep learning - and compared their performances.
    """
    )

    st.header('Methodology')

    st.write(
    """
    For all the models, we proceeded with the following steps:
    1. Split our data into train and test datasets, using 20% of the data for test
    2. Train the model on the train dataset
    3. Make predictions on the test dataset and measure the performance of these predictions
    4. Train the model on the entire dataset
    5. Calculate forecasts for the next 10 years
    """
    )

    st.header('Statistical model : ARIMA')
    st.image('assets/arima.png')
    
    with st.expander("See details"):
        st.write('''
        - MAE: 0.797
        - RMSE: 0.921
        - MAPE: 0.006
        - r2: 0.408\n
        The parameters of the ARIMA model were dermined with a manual grid search:
        ''')

        code = '''p_values = range(0, 6)
d_values = range(0, 3)
q_values = range(0, 6)       

best_score = float("inf")
best_params = (0,0,0)
arima_pred = []

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            warnings.filterwarnings("ignore")
            model = sm.tsa.arima.ARIMA(train, order=order).fit()
            predictions = model.predict(start=len(train), end=len(train) + len(test)-1)
            error = mean_absolute_error(test, predictions)
                if error < best_score:
                    best_score = error
                    best_params = order
                    arima_pred = predictions '''
        
        st.code(code, language='python') 


    st.header('Machine learning models')

    st.subheader('Naive linear regression')
    st.image('assets/naive_lin.png')

    with st.expander("See details"):
        st.write('''
        This model simply uses the years as feature.\n
        - MAE: 5.441
        - RMSE: 5.677
        - MAPE: 0.043
        - r2: -21.482
        ''')
    
    st.subheader('Linear regression')
    st.image('assets/linreg.png')

    with st.expander("See details"):
        st.write('''
        This model uses emissions of year N-1 to predict emissions or year N. \n
        - MAE: 0.939
        - RMSE: 1.317
        - MAPE: 0.007
        - r2: -0.21
        ''')
    
    st.subheader('SVM')
    st.image('assets/svm.png')

    with st.expander("See details"):
        st.write(
            '''
            This model uses emissions of year N-1 to predict emissions or year N and performs a grid search.\n
            - MAE: 0.934
            - RMSE: 1.239
            - MAPE: 0.007
            - r2: -0.072
            ''')

    st.header('Deep learning models')

    st.subheader('Vanilla LSTM')
    st.image('assets/vanilla_lstm.png')

    with st.expander("See details"):

        st.write( '''
            - MAE: 1.202
            - RMSE: 1.461
            - MAPE: 0.01
            - r2: -0.489
            ''')

        code = '''n_steps = 3
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error') 
        '''
        st.code(code, language='python') 

    st.subheader('Stacked LSTM')
    st.image('assets/stacked_lstm.png')
    
    with st.expander("See details"):
        st.write('''
        - MAE: 1.174
        - RMSE: 1.413
        - MAPE: 0.009
        -r2: -0.394
        ''')

        code = '''n_steps = 3
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')
        '''
        st.code(code, language='python') 


    st.subheader('Bidirectional LSTM')
    st.image('assets/bidirectional.png')

    with st.expander("See details"):
        st.write('''
        - MAE: 7.208
        - RMSE: 7.328
        - MAPE: 0.057
        -r2: -36.471
        ''')

        code = '''n_steps = 3
model= Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')
        '''
        st.code(code, language='python') 

    st.subheader('CNN LSTM')
    st.image('assets/cnn_lstm.png')

    with st.expander("See details"):
        st.write('''
        - MAE: 6.08
        - RMSE: 6.282
        - MAPE: 0.048
        - r2: -26.529
        ''')
        code = '''n_steps =4
n_seq = 2
n_steps_seq = 2
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps_seq, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')
        '''
        st.code(code, language='python') 

    st.subheader('Conv SVM')
    st.image('assets/conv_lstm.png')

    with st.expander("See details"):
        st.write('''
        - MAE: 3.635
        - RMSE: 3.892
        - MAPE: 0.029
        - r2: -9.57
        ''')
        code = '''n_steps =4
n_seq = 2
n_steps_seq = 2
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps_seq, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')
        '''
        st.code(code, language='python') 

    st.header('Results')

    st.image('assets/results.png')
    st.image('assets/all_forecasts.png')





                                                    #Page 4: Try it yourself    
        
        
if page == pages[3]:
    
#Selections

 #Country selection   
    country = st.selectbox('Select a Country',options = df.Country.unique(),  index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
   #Sector selection   
    sector = st.selectbox('Select a Sector',options = df.Sector.unique(),  index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
   #Gas type selection
    gas = st.selectbox('Select a gas',options = df.Gas.unique(),  index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
   #Year selection
    year = st.selectbox('Select a Year',options = df.index.unique(),  index=0,  key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")    

   #Corresponding df 
    if st.button('Show Graph'):               

        data = df[(df['Country'] == country) & (df['Sector'] == sector) & (df['Gas'] == gas)]    
        data = data.drop(['Country','Sector','Gas'], axis = 1)
   #plot emissions from selected df        
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.xlabel('Years')
        plt.ylabel('Emissions in MtCo2e')
        st.pyplot(fig)

        
  #download df data as csv
#   @st.cache
#   def convert_df(df):
#       # IMPORTANT: Cache the conversion to prevent computation on every rerun
#       return df.to_csv().encode('utf-8')

#   csv = convert_df(data)

#   st.download_button(
#       label="Download data as CSV",
#       data=csv,
#       file_name='data.csv',
#       mime='text/csv',
#   )

# Plot pie chart of selected datas    
    sectors = ['Agriculture', 'Building', 'Bunker Fuels' ,'Electricity/Heat','Fugitive Emissions', 'Industrial Processes','Manufacturing/Construction','Other Fuel Combustion','Transportation', 'Waste']

    secteurs = df[(df['Sector'].isin(sectors)) &
                              (df.index == year) & 
                              (df['Gas'] == gas) &
                              (df['Country'] == country)]



    fig2, ax = plt.subplots()
    plt.pie(secteurs['cons'], 
            labels = secteurs['Sector'],
            autopct = lambda x:str(round(x,2)) +'%',
            pctdistance = .7)
   
    st.pyplot(fig2)
    
    def calculate_scores(y_true,y_pred):
        print("MAE:", mean_absolute_error(y_true,y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
        print("r2", r2_score(y_true,y_pred))
        
    def Lin_reg(df):
        
        
        df = df.sort_index(ascending = True)
        X_train_lr = pd.DataFrame(df.head(24).index)
        y_train_lr = df.head(24)
        X_test_lr = pd.DataFrame(df.tail(6).index)
        y_test_lr = df.tail(6)

        lr = LinearRegression()
 
        lr.fit(X_train_lr,y_train_lr)
        y_pred_lr = lr.predict(X_test_lr)
        
        
        fig3, ax = plt.subplots()
        plt.plot(data.index, data, label ='reality')
        plt.plot(X_test_lr.values, y_pred_lr, label ='prediction')
        plt.title('Transportation CO2 emissions predictions with Linear Regression')
        plt.ylabel("Emissions in MtCO₂e")
        plt.xticks(rotation=45)
        plt.legend()
    
  
        return st.write('Scores:', calculate_scores(y_test_lr, y_pred_lr)),st.pyplot(fig3)
   
  # Lin_reg(data)
    
    
    def ARIMA(df):
        df = df.sort_index(ascending = True)
        # Finding best parameters for ARIMA model
        df.index= pd.to_datetime(df.index, format='%Y')
        train, test = train_test_split(df, test_size = 0.2, shuffle = False)
        p_values = range(0, 6)
        d_values = range(0, 3)
        q_values = range(0, 6)
    
        best_score = float("inf")
        best_params = (0,0,0)
    
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    warnings.filterwarnings("ignore")
                    model = sm.tsa.arima.ARIMA(train, order=order).fit()
                    predictions = model.predict(start=len(train), end=len(train) + len(test)-1)
                    error = mean_absolute_error(test, predictions)
                    if error < best_score:
                        best_score = error
                        best_params = order
        
 #       # Using best parameters for ARIMA model
 #       
 #       arima = sm.tsa.arima.ARIMA(train, order=best_params)
 #       arima_fitted = arima.fit()
 #          
 #       arima_pred = arima_fitted.get_forecast(steps =6).summary_frame() 
 #         
 #       
 #   # training on full dataset then making forecasts 
 #   
 #       arima = sm.tsa.arima.ARIMA(df, order=best_params)
 #       arima_fitted = arima.fit()
 #   
 #       arima_forecasts = arima_fitted.get_forecast(steps =10).summary_frame() 
        
    # Plot prediction and forecasts
    
    
 #       fig, ax = plt.subplots(figsize = (15,5))
 #       
 #       plt.plot(df, label = 'real emissions')
 #       
 #       arima_pred['mean'].plot(ax = ax, style = 'k--', label = 'predictions') 
 #       
 #       arima_forecasts['mean'].plot(ax = ax, style = '--', color = 'red', label = 'forecasts') 
 #       
 #       ax.fill_between(arima_pred.index, arima_pred['mean_ci_lower'], arima_pred['mean_ci_upper'], color='k', alpha=0.1)
 #       
 #       ax.fill_between(arima_forecasts.index, arima_forecasts['mean_ci_lower'], arima_forecasts['mean_ci_upper'], color='k', alpha=0.1)
 #   
 #       plt.ylabel("Emissions in MtCO₂e")
 #       plt.legend(loc = 'upper left');
 #       
 #       return st.write('Best ARIMA%s MAE=%.3f' % (best_params, best_score))
        return st.write('Best parameters:', best_params)
    
    if st.button('Calculate ARIMA'):
        ARIMA(data)       
   
    
    
if page == pages[4]:
    st.title('Conclusion') 
    st.write(
        """
        Whereas ARIMA and machine learning models have regular performances when they are run several times, we observed that the performances of the deep learning models were quite erratic.
        
        Indeed, deep learning models require a lot of data to perform well.
        
        The limited size of our dataset was the main difficulty in our project: publication of emissions data is quite recent and uses to be on a yearly basis.
        For France, monthly data are only available since 2019. 

        Unfortunately, data augmentation and transfer learning are at an early stage for time series.
        
        """
    )


    