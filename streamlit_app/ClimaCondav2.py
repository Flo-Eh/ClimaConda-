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

# Import libraries for deeplearning
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import Bidirectional
from tensorflow.keras import Sequential
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#Page configuration 
st.set_page_config(page_title="ClimaConda",page_icon='🌍',layout="wide", initial_sidebar_state="auto")

#Barre de Navigation 
st.sidebar.title('Navigation')
#Différentes pages du site 
pages = ['Introduction','Data exploration',' Models','Try it yourself !!!' , 'Conclusion']
page = st.sidebar.radio(' ',pages)

#Importation du df sur l'europe 
df = pd.read_csv('assets/final_df_UE.csv',index_col = 'year')
DF = pd.read_csv('assets/final_df_UE.csv',index_col = 'year')
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
    
                                            
            
    st.markdown("<h2 style='text-align: center;'>Select a Country, Sector, Gas and a Year to do your own predictions</h2>", unsafe_allow_html=True)
    # Selection of Dataframe 
            
    #4 columns for 4 buttons 
    col1, col2, col3, col4 = st.columns(4) 
    
    with col1:  
    #Country selection   
        country = st.selectbox('Select a Country',options = df.Country.unique())
        
    with col2:
    #Sector selection   
        sector = st.selectbox('Select a Sector',options = df.Sector.unique())
        
    with col3:
    #Gas type selection
        gas = st.selectbox('Select a gas',options = df.Gas.unique())
        
    with col4:
    #Year selection
        year = st.selectbox('Select a Year',options = df.index.unique()) 
    


    
    # Parameters for Machine Learning models - Defines shape of window for pre-precessing features and choose model
    window = 5  
    model_name = ["Linear Regression", 'SVM'] 
    
    # Parameters for statistical model ARIMA - set values for p,d,q order
    ###changement des valeurs
    p_values = range(0, 5) 
    d_values = range(0, 3) 
    q_values = range(0, 5)  
    
    
    # Parameters for deep learning configuration 
    deep_learning_config = {'Vanilla LSTM':{"n_step":5},
                             'Stacked LSTM':{"n_step":3},
                             'Bidirectional LSTM':{"n_step":3}
                           }
    cnn_config = {'CNN LSTM':{"n_steps": 4,
                              "n_seq": 2,
                              "n_steps_seq": 2},
                  'Conv LSTM':{"n_steps": 4,
                               "n_seq": 2,
                               "n_steps_seq": 2}
                 }
    
      
    
                                              # charger la fonction de preprocess
    class preprocess_and_evaluate:
        

    
        def __init__(self, country: str, sector: str, gas: str):
            self.country = country
            self.sector = sector
            self.gas = gas
        
        def get_df(country: str, sector: str, gas: str) -> pd.DataFrame:
            """ Get and initialize df """
            
            df = pd.read_csv('final_df_europe.csv')
            df = df.set_index('year')
            df = df[(df['country'] == country) & (df['sector'] == sector) & (df['gas'] == gas)]
            df = df.drop(['Unnamed: 0', 'country', 'sector', 'gas'], axis=1)
            df.index = pd.to_datetime(df.index, format='%Y')
            df = df.sort_values(by='year')
            
            return df
        
        def calculate_scores(y_true: np.array , y_pred: np.array) :
            """ Calculate scores for models : mar, rmse, mape, r2 """
            
            mae = round(mean_absolute_error(y_true,y_pred), 3)
            rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 3)
            mape = round(mean_absolute_percentage_error(y_true, y_pred), 3)
            r2 = round(r2_score(y_true,y_pred), 3)
            
            return mae, rmse, mape, r2
        
        def split_sequence(sequence: np.array, n_steps: int) -> (np.array, np.array):
            """ Split a univariate sequence into samples - Used for deep learning """
            
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence)-1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)
    

        
   

        
                                                    #MACHINE LEARNING      
        
        
    
    class machine_learning_models:


        def __init__(self, window:str):
            self.df = preprocess_and_evaluate.get_df(country, sector, gas)
            self.window = window
    
    
        def window_dataset(self) -> pd.DataFrame:
            """ 
            Defines new features with implementing a new series of df. 
            This series are shifted from orginal dataframe.
            """
            series = self.df
            L = len(series)
            X = series.values
            new_df = pd.DataFrame(data = X)
    
            for i in range(self.window):
                y = []
                for i in range(L - 1):
                    y.append(X[i+ 1])
                new_df = pd.concat([new_df,pd.DataFrame(y)], axis=1)
                L = L-1
                X = y  
            new_df.columns = ['New_col'+str(i) for i in range(self.window+1)]
            
        
            new_df = new_df.iloc[:-self.window]
            new_df.index = series.index[self.window : self.window + self.df.shape[0]] 
            
            return new_df
        
        def split_df(self) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
            """ Use window function and split df with train and test sets"""
            
            self.new_df = self.window_dataset()
            self.target = self.new_df['New_col'+str(self.window)]
            self.data = self.new_df.drop('New_col'+str(self.window), axis = 1)
    
            #training
            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.target,test_size = 0.2, shuffle = False)
            
            return X_train, X_test, y_train, y_test
        
        
        def predict(self, model_name: str) -> (pd.Series, np.array, pd.DataFrame, pd.Series):
            """ Fit and predict models """
            
            X_train, X_test, y_train, y_test = self.split_df()
            
            if model_name == 'Linear Regression':
    
                lr = LinearRegression()
                lr.fit(X_train.values, y_train.values)
    
                #predictions
                y_pred_lr = lr.predict(X_test.values)
    
                #training on full dataset
                lr = LinearRegression()
                lr.fit(self.data.values, self.target.values)
    
                #forecasts
                feat = self.new_df.iloc[-1:,1:self.window+1].values
                
                future = []
                for i in range(11):
                    pred = lr.predict(feat.reshape(1, -1))
                    future.append(pred)
                    feat = np.delete(feat,[0])
                    feat = np.append(feat, pred)
    
                forecasts_lr = pd.Series(data = future, index = pd.to_datetime(range(2020,2031), format='%Y')) 
                
                return y_test, y_pred_lr, X_test, forecasts_lr
    
    
            elif model_name == 'SVM':
    
                params = {'C': np.arange(1,11,10),
                          'kernel': ['linear', 'rbf'],
                          'epsilon': np.arange(0.1, 0.5, 0.1)}
    
                model = SVR()
                cv = GridSearchCV(model, param_grid = params )
                cv.fit(X_train.values, y_train.values)
    
                #predictions
                y_pred_svm = cv.predict(X_test.values)
    
                #training on full dataset
                svm = cv.best_estimator_
                svm.fit(self.data.values,self.target.values)
    
                #forecasts
                feat = self.new_df.iloc[-1:,1:self.window+1].values
    
                future = []
                for i in range(11):
                    pred = svm.predict(feat.reshape(1, -1))
                    future.append(pred)
                    feat = np.delete(feat,[0])
                    feat = np.append(feat,pred)
    
                forecasts_svm = pd.Series(data = future, index = pd.to_datetime(range(2020,2031), format='%Y') )   
    
                return y_test, y_pred_svm, X_test, forecasts_svm  
            
        def show_error(self, model_name: str):
            """ Show errors of different models """
            
            df_score = pd.DataFrame(columns=[])
            
            for name in model_name:
            
                y_test, y_pred, X_test, forecasts = self.predict(name)
                score = preprocess_and_evaluate.calculate_scores(y_test, y_pred)
                
                df_temp = pd.DataFrame(score, columns=[name])
                df_score = pd.concat([df_score, df_temp], axis=1)
                
            fig_error, ax = plt.subplots()    
            barWidth = 0.4
            bars = ('mae', 'rmse', 'mape', 'r2')
            x_pos = np.arange(len(bars))
            x2 = [r + barWidth for r in x_pos]
            position_list = [x_pos, x2]
            
            for name, position in zip(enumerate(model_name), position_list):   
                
                plt.bar(position, df_score[name[1]], width = barWidth, label = name[1])
    
            plt.xticks(x_pos, bars)
            plt.title("Errors")
    
            plt.legend();
            return st.pyplot(fig_error)
        
        def visulalisation_machine_learning(self, model_name: str):
            """ Show visualisation for reality, test predictions and 10 years forecast """
            
            #df = self.df
            #plt.figure(figsize = (10,4))
            
            colors = ['green', 'red']  
            fig, ax = plt.subplots(figsize = (15,5))
            plt.plot(df.index.year, df, label ='reality')
            plt.ylabel("Emissions in MtCO₂e")
            plt.title(f'Predictions and forecasts with Machine Learning model')
            for i, name in enumerate(model_name):
                
                y_test, y_pred, X_test, forecasts = self.predict(name)
                
                plt.plot(X_test.index.year, y_pred, '--', color = colors[i], label = f'prediction for {name}')
                plt.plot(range(2020,2031), forecasts, '-', color = colors[i], label =  f'forecasts for {name}')
                plt.plot(range(2020,2031), forecasts, label =  f'forecasts for {name}')
                
            plt.legend();
            st.pyplot(fig)
        
    

    
        
    class statistics_model:
   
        def __init__(self, p_values: range, d_values: range, q_values: range):
            
            self.df = preprocess_and_evaluate.get_df(country, sector, gas)
            self.train, self.test = train_test_split(self.df, test_size = 0.2, shuffle = False)
            self.p_values = p_values
            self.d_values = d_values
            self.q_values = q_values
            
        def best_order_arima(self) -> (tuple, float):
            """ Find best value order for p, d and q """
            
            best_score, best_params = float("inf"), None
            for p in self.p_values:
                for d in self.d_values:
                    for q in self.q_values:
                        order = (p,d,q)
                        warnings.filterwarnings("ignore")
                        model = sm.tsa.arima.ARIMA(self.train, order=order).fit()
                        predictions = model.predict(start=len(self.train), end=len(self.train) + len(self.test)-1)
                        error = mean_absolute_error(self.test, predictions)
                        if error < best_score:
                            best_params = order
                            best_score = error
       
            return best_params, best_score
        
        def predict_arima(self) -> (pd.DataFrame, pd.DataFrame):
            """ Fit and predict ARIMA model """
            
            # Fit Arima for predictions
            best_params = self.best_order_arima()[0]
            arima = sm.tsa.arima.ARIMA(self.train, order=best_params)
            arima_fitted = arima.fit()
            arima_pred = arima_fitted.get_forecast(steps=6).summary_frame() 
    
            # Fit with entire dataset
            arima = sm.tsa.arima.ARIMA(self.df, order=best_params)
            arima_fitted = arima.fit()
    
            # Forecast for 10 years
            arima_forecasts = arima_fitted.get_forecast(steps=10).summary_frame() 
            
    
            return arima_pred, arima_forecasts
    
        
        def show_error(self):
            """ Show error ARIMA model """
            
            # Get predictions and Forecast
            arima_pred, arima_forecasts = self.predict_arima()
            
            # Calculate score
            score = preprocess_and_evaluate.calculate_scores(self.test, arima_pred['mean'])
            df_score = pd.DataFrame(score, columns=['arima'])
    
            # Display results
            fig_error, ax = plt.subplots()
            barWidth = 0.4
            bars = ('mae', 'rmse', 'mape', 'r2')
            x_pos = np.arange(len(bars))
            plt.bar(x_pos, df_score['arima'], width = barWidth, label = 'arima errors')
            plt.xticks(x_pos, bars)
            plt.title("Errors")
            plt.legend();    
            st.pyplot(fig_error)
        
        def visualisation_arima_model(self):
            """ Show visualisation for reality, test predictions and 10 years forecast """
            
            # Get predictions and Forecast
            arima_pred, arima_forecasts = self.predict_arima()
            
            # Display results
            fig, ax = plt.subplots(figsize = (15,5))
            plt.plot(self.df, label = 'real emissions')
            arima_pred['mean'].plot(ax = ax, style = 'k--', label = 'predictions') 
            arima_forecasts['mean'].plot(ax = ax, style = '--', color = 'red', label = 'forecasts') 
            ax.fill_between(arima_pred.index, arima_pred['mean_ci_lower'], 
                            arima_pred['mean_ci_upper'], color='k', alpha=0.1)
            ax.fill_between(arima_forecasts.index, arima_forecasts['mean_ci_lower'], 
                            arima_forecasts['mean_ci_upper'], color='k', alpha=0.1)
    
            plt.ylabel("Emissions in MtCO₂e")
            plt.legend(loc = 'upper left');
            st.pyplot(fig)
            
            
            
            

                                                        # Deep Learning         
                
                
    class deep_learning_models:
        """ 
        Preprocess, fit, predit, show errors and visualisation for deep learning models : 
        Vanilla LSTM, Stacked LSTM, Bidirectional LSTM

        """ 

        def __init__(self):
            self.df = preprocess_and_evaluate.get_df(country, sector, gas)
            self.deep_learning_config = deep_learning_config


        def fit_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, X: np.array, y: np.array, 
                      model_name: str, n_steps: int, n_features: int) -> (np.array, np.array):
            """ Fit and predict models """

            if model_name == 'Vanilla LSTM':

                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_absolute_error')

                # fit model
                model.fit(X, y, epochs=200, verbose=0)   

                #predictions
                n_years = self.df.shape[0] - X.shape[0] + X_test.shape[0]
                Xinput = X[-1]

                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                predictions = Xinput[-X_test.shape[0]:]  

                #reset model and train on full dataset
                model.reset_states()

                X, y = preprocess_and_evaluate.split_sequence(self.df.values, n_steps)
                n_features = 1
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                model.fit(X, y, epochs=200, verbose=0)

                #forecasts
                Xinput = X[-1]
                n_years = self.df.shape[0] - X.shape[0] + 11
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                forecasts = Xinput[-11:] 

                return predictions, forecasts

            if model_name == 'Stacked LSTM':

            # define model
                model = Sequential()
                model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
                model.add(LSTM(50, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_absolute_error')
                # fit model
                model.fit(X, y, epochs=200, verbose=0)

                #predictions
                n_years = self.df.shape[0] - X.shape[0] + X_test.shape[0]
                Xinput = X[-1]

                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                predictions = Xinput[-X_test.shape[0]:]  

                #reset model and train on full dataset
                model.reset_states()

                X, y = preprocess_and_evaluate.split_sequence(self.df.values, n_steps)
                n_features = 1
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                model.fit(X, y, epochs=200, verbose=0)

                #forecasts
                Xinput = X[-1]
                n_years = self.df.shape[0] - X.shape[0] + 11
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                forecasts = Xinput[-11:] 

                return predictions, forecasts

            if model_name == 'Bidirectional LSTM':

                # define model
                model= Sequential()
                model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_absolute_error')

                # fit model
                model.fit(X, y, epochs=200, verbose=0)

                #predictions
                n_years = self.df.shape[0] - X.shape[0] + X_test.shape[0]
                Xinput = X[-1]

                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                predictions = Xinput[-X_test.shape[0]:]  

                #reset model and train on full dataset
                model.reset_states()

                X, y = preprocess_and_evaluate.split_sequence(self.df.values, n_steps)
                n_features = 1
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                model.fit(X, y, epochs=200, verbose=0)

                #forecasts
                Xinput = X[-1]
                n_years = self.df.shape[0] - X.shape[0] + 11
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_steps, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))

                forecasts = Xinput[-11:]

                return predictions, forecasts

        def get_error(self) -> (pd.DataFrame, np.array, np.array, pd.DataFrame):
            """ Calculate error for deeplearning models """

            df_score = pd.DataFrame(columns=[])
            X_train, X_test = train_test_split(self.df, test_size = 0.2, shuffle = False)

            for model_name, feat in self.deep_learning_config.items():

                n_features = 1
                X, y = preprocess_and_evaluate.split_sequence(X_train.values, feat['n_step'])
                X = X.reshape((X.shape[0], X.shape[1], n_features))

                predictions, forecasts = self.fit_model(X_train, X_test, X, y, model_name, 
                                                   feat['n_step'], n_features)

                score = preprocess_and_evaluate.calculate_scores(X_test, predictions)           
                df_temp = pd.DataFrame(score, columns=[model_name])
                df_score = pd.concat([df_score, df_temp], axis=1)

            return X_test, predictions, forecasts, df_score


        def show_error(self):
            """ Visualise error for models """

            X_test, predictions, forecasts, df_score = self.get_error()
            fig_error, ax = plt.subplots(figsize = (15,5))
            barWidth = 0.3
            bars = ('mae', 'rmse', 'mape', 'r2')
            x_pos = np.arange(len(bars))
            x2 = [r + barWidth for r in x_pos]
            x3 = [r + barWidth for r in x2]
            position_list = [x_pos, x2, x3]

            for model_name, position in zip(enumerate(self.deep_learning_config.items()), position_list):

                plt.bar(position, df_score[list(deep_learning_config.keys())[model_name[0]]], 
                        width = barWidth, label = list(deep_learning_config.keys())[model_name[0]])

            plt.title("Errors")
            plt.xticks(x_pos, bars)
            plt.legend()
            st.pyplot(fig_error);


        def dl_model_visualisation(self):
            """ Show visualisation for reality, test predictions and 10 years forecast """

            #Plot graph   
            fig, ax = plt.subplots(figsize = (15,5))
            plt.plot(self.df.index.year, self.df, label ='Reality')
            plt.xlabel('Years')
            plt.ylabel('Emissions in MtCO2e')
            plt.title('Predictions and forecasts')

            for model_name, feat in self.deep_learning_config.items():

                #Get variables for predictions 
                X_test, predictions, forecasts, df_score = self.get_error() 

                #Plot predictions
                plt.plot(X_test.index.year, predictions ,label= f'Prediction {model_name}')
                plt.plot(range(2020, 2031), forecasts, label = f'Forecasts {model_name}')

            plt.legend()
            st.pyplot(fig);    
            
            
            
            
                                                    #CNN models 

                
                
                
                
    class cnn_models:
        """ 
        Preprocess, fit, predit, show errors and visualisation for Convolutional Neural Networks : 
        CNN LSTM, Conv LSTM
        
        """    
        
        def __init__(self):
            self.df = preprocess_and_evaluate.get_df(country, sector, gas)
            self.cnn_config = cnn_config
    
        def fit_model(self, X_train: pd.DataFrame, X_test:pd.DataFrame, X: np.array, y: np.array, model_name: str, 
                      n_steps: int, n_seq: int , n_steps_seq: int, n_features: int) -> (np.array, np.array) :
            """ Fit and predict models """
            
            if model_name == 'CNN LSTM':
    
                model = Sequential()
                model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps_seq, n_features)))
                model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
                model.add(TimeDistributed(Flatten()))
                model.add(LSTM(50, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_absolute_error')
                # fit model
    
                model.fit(X, y, epochs=500, verbose=0)
    
                #predictions
                n_years = self.df.shape[0] - X.shape[0] + X_test.shape[0]
                Xinput = X[-1].reshape(-1,1)
    
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_seq, n_steps_seq, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))
    
                predictions = Xinput[-X_test.shape[0]:]  
    
                #reset model and train on full dataset
                model.reset_states()
    
                X, y = preprocess_and_evaluate.split_sequence(self.df.values, n_steps)
                n_features = 1
                X = X.reshape((X.shape[0], n_seq, n_steps_seq, n_features))
    
                model.fit(X, y, epochs=200, verbose=0)
    
                #forecasts
                Xinput = X[-1].reshape(-1,1)
                n_years = self.df.shape[0] - X.shape[0] + 11
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_seq, n_steps_seq, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))
    
                forecasts = Xinput[-11:] 
    
                return predictions, forecasts
            
            if model_name == 'Conv LSTM':
                
                # define model
                model = Sequential()
                model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps_seq, n_features)))
                model.add(Flatten())
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_absolute_error')
    
                # fit model
                model.fit(X, y, epochs=500, verbose=0) 
    
                #predictions
                n_years = self.df.shape[0] - X.shape[0] + X_test.shape[0]
                Xinput = X[-1].reshape(-1,1)
    
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_seq, 1, n_steps_seq, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))
    
                predictions = Xinput[-X_test.shape[0]:]  
    
                #reset model and train on full dataset
                model.reset_states()
                
                n_features = 1
                X, y = preprocess_and_evaluate.split_sequence(self.df.values, n_steps)
                X = X.reshape((X.shape[0], n_seq, 1, n_steps_seq, n_features))
    
                model.fit(X, y, epochs=200, verbose=0)
    
                #forecasts
                Xinput = X[-1].reshape(-1,1)
                n_years = self.df.shape[0] - X.shape[0] + 11
                
                for i in range(n_years):
                    x_input = Xinput[-n_steps:]
                    x_input = x_input.reshape((1, n_seq, 1, n_steps_seq, n_features))
                    y = model.predict(x_input, verbose=0)
                    Xinput = np.concatenate((Xinput, y))
    
                forecasts = Xinput[-11:] 
    
                return predictions, forecasts
            
        def get_error(self) -> (pd.DataFrame, np.array, np.array, pd.DataFrame):
            """ Calculate error for cnn models """
            
            df_score = pd.DataFrame(columns=[])
     
            for model_name, feat in self.cnn_config.items():
    
                if not ((feat['n_steps'] >= feat['n_seq']) and (feat['n_seq'] >= feat['n_steps_seq'])):
                    print('Paramètres incompatibles : veuillez choisir n_steps >= n_seq >= n_steps_seq')
    
                else :         
                    X_train, X_test = train_test_split(self.df, test_size = 0.2, shuffle = False)                   
    
                    # split into samples
                    X, y = preprocess_and_evaluate.split_sequence(X_train.values, feat['n_steps'])
    
                   # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
                    n_features = 1
                    if model_name == 'CNN LSTM':
    
                        X = X.reshape((X.shape[0], feat['n_seq'], feat['n_steps_seq'], n_features))
    
                    elif model_name == 'Conv LSTM':
    
                        X = X.reshape((X.shape[0], feat['n_seq'], 1, feat['n_steps_seq'], n_features))
    
                    predictions, forecasts = self.fit_model(X_train, X_test, X, y, model_name, feat['n_steps'], 
                                                            feat['n_seq'], feat['n_steps_seq'], n_features)
    
                    score = preprocess_and_evaluate.calculate_scores(X_test, predictions)
                    df_temp = pd.DataFrame(score, columns=[model_name])
                    df_score = pd.concat([df_score, df_temp], axis=1)
                
            return X_test, predictions, forecasts, df_score
            
        def show_error(self):
            """ Visualise error for cnn models """
    
            X_test, predictions, forecasts, df_score = self.get_error()
            fig_error, ax = plt.subplots(figsize = (15,5))        
            barWidth = 0.4
            bars = ('mae', 'rmse', 'mape', 'r2')
            x_pos = np.arange(len(bars))
            x2 = [r + barWidth for r in x_pos]
            position_list= [x_pos, x2]
            
            for model_name, position in zip(enumerate(self.cnn_config.items()), position_list):
                
                plt.bar(position, df_score[list(cnn_config.keys())[model_name[0]]], 
                        width = barWidth, label = list(cnn_config.keys())[model_name[0]])
    
            plt.xticks(x_pos, bars)
            plt.legend()
            st.pyplot(fig_error);
            
    
        def cnn_model_visualisation(self):
            """ Show visualisation for reality, test predictions and 10 years forecast """
            
            # Plot graph
            fig, ax = plt.subplots(figsize = (15,5))
            plt.plot(self.df.index.year, self.df, label ='Reality')
            plt.xlabel('Years')
            plt.ylabel('Emissions in MtCO2e')
            plt.title('Predictions and forecasts')
    
            for model_name, feat in self.cnn_config.items():
    
                X_test, predictions, forecasts, df_score = self.get_error()
    
                plt.plot(X_test.index.year, predictions ,label= f'Prediction {model_name}')
                plt.plot(range(2020, 2031), forecasts, label = f'Forecasts {model_name}')
    
            plt.legend()
            plt.show()
            st.pyplot(fig); 
                
                
                
                
                
                
                                        # Adding Buttons for everything    
                    
                    
                    
                    
                    
    
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
    st.markdown("<h2 style='text-align: center;'>Data exploration</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
        )  
            
     
        
    with col2:  
        
    # Instanciate, show errors and forecast
        df = preprocess_and_evaluate.get_df(country,sector,gas)

        if st.button('Show corresponding DataFrame'):      
            st.dataframe(df)
            
    st.markdown("<h2 style='text-align: center;'>Data visualization</h2>", unsafe_allow_html=True)        
    col1, col2 = st.columns(2)        
    with col1:
            # plot emissions over time for selected data 
        if st.button('Show Graph'):
            fig_general, ax = plt.subplots()
            ax.plot(df)
            plt.xlabel('Years')
            plt.ylabel('Emissions in MtCo2e')
            st.pyplot(fig_general)
    with col2:
        # Plot pie chart of selected datas    
        sectors = ['Agriculture', 'Building', 'Bunker Fuels' ,'Electricity/Heat','Fugitive Emissions', 'Industrial Processes','Manufacturing/Construction','Other Fuel Combustion','Transportation', 'Waste']
    
        secteurs = DF[(DF['Sector'].isin(sectors)) &
                                  (DF.index == year) & 
                                  (DF['Gas'] == gas) &
                                  (DF['Country'] == country)]
    
        if True in list(secteurs.isna().any()):
            st.write('Cannot compute models : the series contains NaN values')
        else:
    
            fig_pie, ax = plt.subplots()
            plt.pie(secteurs['cons'], 
                    labels = secteurs['Sector'],
                    autopct = lambda x:str(round(x,2)) +'%',
                    pctdistance = .7)
            
            if st.button('Show Repartition of emissions over the different sectors for a specified year, gas and country'):    
                st.pyplot(fig_pie)
        
    st.markdown("<h2 style='text-align: center;'>Machine Learning</h2>", unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1,2])
    with col1:
        model_select = st.multiselect('Select a Model',options = model_name  )  
    with col2:
        error_ML = st.checkbox('Display Machine Learning Errors')
        
        if error_ML:
            
            if st.button('Machine Learning Prediction and Forecasts '):
                stat = machine_learning_models(window=window)
                stat.show_error(model_name=model_select )
                stat.visulalisation_machine_learning(model_name=model_select )
        else:
            if st.button('Machine Learning Prediction and Forecasts '):
                stat = machine_learning_models(window=window)
                #stat.show_error(model_name=model_select )
                stat.visulalisation_machine_learning(model_name=model_select )
    
    st.markdown("<h2 style='text-align: center;'>Statistical Model</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)                    
                   
    with col2:               
        if st.button('ARIMA (may take a few seconds) '):        
            stat = statistics_model(p_values=p_values, d_values=d_values, q_values=q_values)
            stat.show_error()
            stat.visualisation_arima_model()
            
    st.markdown("<h2 style='text-align: center;'>Deep Learning </h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,2])
    with col1:
        model_deep = st.multiselect('Select Deep Learning Models',options = deep_learning_config.keys() )
        step_select = st.selectbox('Select step value',options = range(1,5) )
        if len(model_deep) == 0:
            print('nothing')
        if len(model_deep) == 1:
            deep_learning_config = {model_deep[0]:{"n_step":step_select}                            
                               }
        if len(model_deep) == 2:
            deep_learning_config = {model_deep[0]:{"n_step":step_select},
                                 model_deep[1]:{"n_step":step_select}                            
                               }
        
        if len(model_deep) == 3:
            deep_learning_config = {model_deep[0]:{"n_step":step_select},
                                 model_deep[1]:{"n_step":step_select},
                                 model_deep[2]:{"n_step":step_select}
                               }
        
     
    
    with col2:
        error_DL = st.checkbox('Display Deep Learning Errors')
        
        if error_DL:
            if st.button('Deep Learning '):
                dl = deep_learning_models()
                dl.show_error()
                dl.dl_model_visualisation()
        else:
            if st.button('Deep Learning '):
                dl = deep_learning_models()
                
                dl.dl_model_visualisation()
            

    st.markdown("<h2 style='text-align: center;'>CNN Learning </h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,2])
    with col1:
        model_cnn = st.multiselect('Select CNN Deep Learning Models',options = cnn_config.keys() )
        if len(model_cnn) == 0:
            print('nothing')
        if len(model_cnn) == 1:
            cnn_config = {model_cnn[0]:{"n_steps": 4,
                          "n_seq": 2,
                          "n_steps_seq": 2}                            
                               }
        if len(model_cnn) == 2:
            cnn_config = {model_cnn[0]:{"n_steps": 4,
                          "n_seq": 2,
                          "n_steps_seq": 2},
                                 model_cnn[1]:{"n_steps": 4,
                          "n_seq": 2,
                          "n_steps_seq": 2}                            
                              }
        
       
     
    
    with col2:
        error_CNN = st.checkbox('Display CNN Errors')
        
        if error_CNN:
            if st.button('CNN Learning '):
                dl = cnn_models()
                dl.show_error()
                dl.cnn_model_visualisation()
        else:
            if st.button('CNN Learning '):
                dl = cnn_models()                
                dl.cnn_model_visualisation()

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


    