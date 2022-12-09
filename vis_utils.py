import datetime
import gc
import logging
import matplotlib as mpl
import numpy as np
import numpy as np
import os
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sys
import tensorflow as tf
import time
import warnings


from copy import deepcopy
from copy import deepcopy
from dataclasses import dataclass
from lightgbm import LGBMRegressor
from math import sqrt
from matplotlib import pyplot as plt
from pycaret.regression import *
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


class DataStore():

    def __init__(self) -> None:
        self.load_data()

        # align matplotlib object 
        mpl.rc('xtick', labelsize=12) 
        mpl.rc('ytick', labelsize=12) 
        mpl.rc('axes', titlesize=18)

    def load_data(self):
        df = pd.read_csv('./data/mice_final.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df[df.groupby('id')['id'].transform('count')>50]
        
        self.df = df

    def build_dataset(self, **kwargs):
        logging.info(f"{id(self)}, running build_dataset") 

        self.__dict__.update(kwargs)
        self.date_ = [pd.to_datetime(date) for date in self.date_]

        df = self.df
        
        datasets = {}
        by_class = df.groupby('id')

        for groups, data in by_class:
            datasets[groups] = data

            
        datasets2 = {}
        by_class = df.groupby('id')

        for groups, data in by_class:
            datasets2[groups] = data
            
            
        unique_id = df['id'].unique()

        target_list = [self.target_]
        a = self.variable_ + target_list

        for i in self.id_:
            datasets[i] = datasets[i][a]
            datasets[i] = datasets[i].loc[(datasets[i].index > self.date_[0]) & (datasets[i].index < self.date_[1])]
            
        target_list = [self.target_]
        a = self.variable_ + target_list

        # save to datastore
        self.datasets = datasets
        self.datasets2 = datasets2
        self.unique_id = unique_id
        self.a = a

        logging.info(f"{id(self)}, completed build_dataset")

    def plot_variable(self):

        logging.info(f"{id(self)}, running plot_variable")
        datasets = self.datasets
        figs = []
        for i in self.id_:
            variables = self.a
            fig = go.Figure()

            for variable in variables:
                fig.add_traces(go.Scatter(x=datasets[i].index, y=datasets[i][variables], mode='lines', name = variable))
                fig.update_layout(title='Plot for all the selected variables ' + i ,
                        xaxis_title='Date',
                        yaxis_title=variable)

            figs.append(fig)
        logging.info(f"{id(self)}, completed plot_variable")
        return figs

    def plot_correlation_matrix(self):
        
        logging.info(f"{id(self)}, running plot_correlation_matrix")
        datasets = self.datasets
        figs = []
        for i in self.id_:

            df_corr = datasets[i].corr() # Generate correlation matrix

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x = df_corr.columns,
                    y = df_corr.index,
                    z = np.array(df_corr)
                )
            )
            
            figs.append(fig)
        logging.info(f"{id(self)}, completed plot_correlation_matrix")
        return figs

    def plot_facet_grid(self):

        logging.info(f"{id(self)}, running plot_facet_grid")
        datasets2 = self.datasets2
        a = self.a
        id_ = self.id_

        for i in id_:
            datasets2[i]['id'] = i

        df2 = datasets2[id_[0]]
        
        for i in id_[1:]:
            df2 = pd.concat([df2, datasets2[i]], axis=0) 
            
        list_district = list(set(df2.id))

        df2['year'] = df2.index.year
        df2['month'] = df2.index.month

        figs = []
        for c in a:
            g = sns.FacetGrid(df2, col="month", row="year", height=3.5, aspect=1)
            g = g.map(sns.barplot, 'id', c, palette='viridis', ci=None, order = list_district)

            g.set_xticklabels(rotation = 90)
            # plt.show()
            figs.append(g)

        self.df2 = df2
        logging.info(f"{id(self)}, completed plot_facet_grid")
        return figs
    
    def plot_catplot(self):
        logging.info(f"{id(self)}, running plot_catplot")
        a = self.a
        df2 = self.df2
        figs = []
        for c in a:
            g = sns.catplot(x='month', y=c, data=df2, col='id', kind='boxen')
            figs.append(g)
        logging.info(f"{id(self)}, completed plot_catplot")
        return figs

    def plot_kde_plot(self):
        logging.info(f"{id(self)}, running plot_kde_plot")
        a = self.a
        df2 = self.df2

        cluster_variables = a

        df3 = df2.set_index("id")
        df3 = df3.stack()
        df3 = df3.reset_index()

        df3 = df3.rename(
            columns={"level_1": "Attribute", 0: "Values"}
        )

        sns.set(font_scale=1.5)
        # Setup the facets
        facets = sns.FacetGrid(
            data=df3,
            col="Attribute",
            hue="id",
            sharey=False,
            sharex=False,
            aspect=2,
            col_wrap=3,
        )
        # Build the plot from `sns.kdeplot`
        g = facets.map(sns.kdeplot, "Values", shade=True).add_legend()
        
        logging.info(f"{id(self)}, completed plot_kde_plot")
        return g

    def plot_corr_dist_plot(self):

        logging.info(f"{id(self)}, running plot_corr_dist_plot")

        def corrdot(*args, **kwargs):
            corr_r = args[0].corr(args[1], 'pearson')
            corr_text = f"{corr_r:2.2f}".replace("0.", ".")
            ax = plt.gca()
            ax.set_axis_off()
            # marker_size = abs(corr_r) * 10000
            ax.scatter([.5], [.5], c=[corr_r], alpha=0.6, cmap="coolwarm",
                    vmin=-1, vmax=1, transform=ax.transAxes)
            font_size = abs(corr_r) * 40 + 5
            ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                        ha='center', va='center', fontsize=font_size)

        target_ = self.target_
        id_ = self.id_

        cluster_variable = target_
        datasets = self.datasets
        
        figs = []
        for i in id_:
            sns.set(style='white', font_scale=1)
            g = sns.PairGrid(datasets[i], aspect=1.8, diag_sharey=False)
            g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
            g.map_diag(sns.distplot, kde_kws={'color': 'black'})
            g.map_upper(corrdot)
            g.fig.suptitle('Correlation distrubution plot - Well ID :' + i)
            figs.append(g)
        logging.info(f"{id(self)}, completed plot_corr_dist_plot")
        return figs
   
    def plot_prediction(self, st_tab):
        
        logging.info(f"{id(self)}, running plot_prediction")
        id_, target_ = self.id_, self.target_
        datasets = self.datasets

        

        for i in id_:

            st_tab.write(f"Prediction for Id = {i}")

            df = datasets[i]
            x = df.drop(target_,axis=1)
            y = df[target_]

            #spliting the dataset into 85:15 ratio
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
            names = df.columns
            
            
            s = setup(data = df, target = target_, session_id=123, silent=True)
            best = compare_models()
            
            leaderboard = get_leaderboard()
            st_tab.dataframe(leaderboard)

            # layout for plot
            container = st_tab.container()
            col1, col2 = container.columns(2)
            
            # Fitting optimized GBDT Regression to the entire data
            # Say, "the default sans-serif font is COMIC SANS"
            
            # mpl.rcParams['font.sans-serif'] = 'Times New Roman'
            # label_size = 20
            # mpl.rcParams['xtick.labelsize'] = label_size 
            # mpl.rcParams['ytick.labelsize'] = label_size 
            
            model =  GradientBoostingRegressor(learning_rate= 0.05, max_depth = 4, min_samples_leaf = 1, min_samples_split = 2, 
                                            n_estimators = 300, subsample = 0.5, random_state=42)

            model.fit(x_train, y_train)
            
            
            print("Prediction for Well ID: %s" % i)

            #Measure the R2 for training and test set
            model_score = model.score(x_train,y_train)
            print("The training R2 is: %.3f" % model.score(x_train, y_train))
            print("The test R2 is: %.3f "% model.score(x_test, y_test))



            y_predicted = model.predict(x_test)
            
            print("MSE: %.3f"% mean_squared_error(y_test, y_predicted))

            # The mean squared error & Variance
            print("MSE: %.3f"% mean_squared_error(y_test, y_predicted))
            print("RMSE: %.3f"% sqrt(mean_squared_error(y_test, y_predicted)))


            #k-cross validation
            
            accuracies = cross_val_score(estimator = model, X = x_train, y= y_train, cv=5)
            print("The mean training accuracy is: %.3f"% accuracies.mean())

            #Plotting the joint plot of  actual v/s predicted
            pp_tr = model.predict(x_train)
            
            print("", end = "\n")
            
            sc = model.score(x_test, y_test)

            fig = px.scatter(df, x=y_train, y=pp_tr, trendline="ols")
            fig.update_layout(xaxis_title = 'Actual Target Hexavalent Chromium: R2 =' + str(sc) + ' for Well ID: ' + i, yaxis_title = 'Predicted Target:  Gradient Boosting')

            # fig.show()

            col1.plotly_chart(fig)

            print("", end = "\n")
            
            lightgbm = create_model('lightgbm')
            with col2:
                plot_model(lightgbm, plot='feature', scale=0.6, display_format="streamlit")
            print("", end = "\n")
            # evaluate_model(lightgbm)

        logging.info(f"{id(self)}, completed plot_prediction")
            
    def visualize_forecast(self, st_tab):
        
        logging.info(f"{id(self)}, running visualize_forecast")

        def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
            data = []
            labels = []

            start_index = start_index + history_size
            if end_index is None:
                end_index = len(dataset) - target_size

            for i in range(start_index, end_index):
                indices = range(i-history_size, i, step)
                data.append(dataset[indices])

                if single_step:
                    labels.append(target[i+target_size])
                else:
                    labels.append(target[i:i+target_size])

            return np.array(data), np.array(labels)

        def multi_step_plot(history, true_future, prediction):
            fig = plt.figure(figsize=(18, 6))
            num_in = create_time_steps(len(history))
            num_out = len(true_future)

            plt.plot(num_in, np.array(history[:, 1]), label='History')
            plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
                label='True Future')
            if prediction.any():
                plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                        label='Predicted Future')
            plt.legend(loc='upper left')
            plt.ylabel('Hexavalent Chromium')
            plt.xlabel('Time-Step')
            # plt.show()
            st_tab.pyplot(fig)
            
        def create_time_steps(length):
            return list(range(-length, 0))

        def show_plot(plot_data, delta, title):
            labels = ['History', 'True Future', 'Model Prediction']
            marker = ['.-', 'rx', 'go']
            time_steps = create_time_steps(plot_data[0].shape[0])
            if delta:
                future = delta
            else:
                future = 0

            plt.title(title)
            for i, x in enumerate(plot_data):
                if i:
                    plt.plot(future, plot_data[i], marker[i], markersize=10,
                        label=labels[i])
                else:
                    plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
                plt.legend()
                plt.xlim([time_steps[0], (future+5)*2])
                plt.xlabel('Time-Step')
            
            return plt

        def plot_train_history(history, title):
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(len(loss))

            plt.figure()

            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title(title)
            plt.legend()

            plt.show()

        id_, target_, variable_ = self.id_, self.target_, self.variable_
        datasets = self.datasets

        for i in id_:
            df = datasets[i]
            df.sort_index(inplace=True)
            x = df.drop(target_,axis=1)
            y = df[target_]
            
            start_time = df.index[0]

            end_time = df.index[-1]
            logging.info(f"{id(self)}, visualize_forecast, {start_time, end_time}")
            start_times = pd.date_range(start= pd.Timestamp(start_time).floor('10T'), end = pd.Timestamp(end_time).floor('10T'), freq="10T")

            end_times = start_times + pd.Timedelta('10T')

            start_times_list = start_times.strftime('%Y-%m-%d %H:%M:%S').tolist()
            logging.info(f"{id(self)}, visualize_forecast, {start_times}, start_times_list")
            len1 = len(start_times_list)
            data_array1 = np.array(df.T.values)

            new_ts = TimeSeriesResampler(sz=len1).fit_transform(data_array1)
            data_array1 = np.squeeze(new_ts)
            
            target_list = [target_]
            a = variable_ + target_list
            
            ll=[]
            for j in range(len1):
                l = []
                for i in range(len(a)):
                    l.append(data_array1[i][j])
                ll.append(l)
            data_array = np.array(ll)
            df = pd.DataFrame(data_array, columns=a, index=start_times_list)
            
            # mpl.rcParams['figure.figsize'] = (17, 5)
            # mpl.rcParams['axes.grid'] = False
            sns.set_style("whitegrid")

            # Data Loader Parameters
            BATCH_SIZE = 256
            BUFFER_SIZE = 1000
            TRAIN_SPLIT = 11344

            # LSTM Parameters
            EVALUATION_INTERVAL = 200
            EPOCHS = 2
            PATIENCE = 5

            # Reproducibility
            SEED = 13
            tf.random.set_seed(SEED)
            
            features = df[a]
            dataset = features.values
            
            past_history = 144
            future_target = 18 # FIXME: 24 to 18
            STEP = 1

            x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                            TRAIN_SPLIT, past_history,
                                                            future_target, STEP)
            x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                        TRAIN_SPLIT, None, past_history,
                                                        future_target, STEP)
            
            train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
            train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

            val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
            val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
            
            multi_step_model = tf.keras.models.Sequential()
            multi_step_model.add(tf.keras.layers.LSTM(32,
                                                    return_sequences=True,
                                                    input_shape=x_train_multi.shape[-2:]))
            multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
            multi_step_model.add(tf.keras.layers.Dense(18))
            multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
            print(multi_step_model.summary())
            
            early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)
            multi_step_history = multi_step_model.fit(train_data_multi,
                                                    epochs=2,
                                                    steps_per_epoch=EVALUATION_INTERVAL,
                                                    validation_data=val_data_multi,
                                                    validation_steps=EVALUATION_INTERVAL,
                                                    callbacks=[early_stopping])

            print("", end = "\n")
            
            for x, y in val_data_multi.take(3):
                multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
            
        logging.info(f"{id(self)}, completed visualize_forecast")

    def visualize_forecast_cluster(self, st_tab):
        
        logging.info(f"{id(self)}, running visualize_forecast_cluster")

        id_, cluster_variable_, cluster_number_, a = self.id_, self.cluster_variable_, self.cluster_number_, self.a
        datasets = self.datasets

        N = len(id_)

        for c in a:
            if(c==cluster_variable_):
                datasetx = datasets
                for i in id_:
                    datasetx[i] = datasetx[i][[c]]
                    print(datasetx[i])
                    print('------')

                l = []
                    
                for i,j in enumerate(id_):
                    globals()[f'df_{i}'] = datasetx[j]
                    
                for i,j in enumerate(id_):
                    len_ = len(df_0)
                    len_1 = len(globals()[f'df_{i}'])
                    len_ = max(len_,len(globals()[f'df_{i}']))
                    
                for i,j in enumerate(id_):
                    globals()[f'data_array_{i}'] = np.array(globals()[f'df_{i}'].T)

                    globals()[f'd_array_{i}'] = []
                    for a in globals()[f'data_array_{i}']:
                        for b in a:
                            globals()[f'd_array_{i}'].append(b)
                

                    print('*****************')
                    
                    globals()[f'd_array_{i}'] = np.array(globals()[f'd_array_{i}'])
                    print(type(globals()[f'd_array_{i}']))
                    new_ts = TimeSeriesResampler(sz=len_).fit_transform(globals()[f'd_array_{i}'])
                    globals()[f'd_array_{i}'] = np.squeeze(new_ts)
                    globals()[f'd_array_{i}'] = globals()[f'd_array_{i}'].tolist()
                
                    l.append(globals()[f'd_array_{i}'])
                        
                    
                        
        # cities_list = l

        my_array = np.array(l)
        model = TimeSeriesKMeans(n_clusters=cluster_number_, metric="dtw", max_iter=20)
        model.fit(my_array)
        # cities_list = data.T.index.tolist()

        y=model.predict(my_array)  
        cluster_id = list(set(y))
        c = cluster_variable_

        for i,j in enumerate(id_):
            globals()[f'df_{i}'] = globals()[f'df_{i}'].rename(columns={c:c+'_'+j+'_cluster-'+ str(y[i])})

        for i in range(len(id_)):
            if(len(globals()[f'df_{i}'].index) == len_):
                df_global = globals()[f'df_{i}']

                
        for i,j in enumerate(y):
            df_global[globals()[f'df_{i}'].columns[0]] = globals()[f'd_array_{i}']
            
        df_global.reset_index(inplace = True)

        for i in cluster_id:
            col_list = [col for col in df_global.columns if col.endswith(str(i))]
            col_list.append('time')
            print(col_list)
            globals()[f'df_cluster_{i}'] = df_global[col_list]
            
        
        cluster_id = list(set(y))

        for i in cluster_id:
            print('Cluster ID ' + str(i) )
            
            fig = px.line( globals()[f'df_cluster_{i}'], x='time', y= globals()[f'df_cluster_{i}'].columns,
                hover_data={"time": "|%B %d, %Y"},
                title='Cluster Number ' + str(i))
            fig.update_xaxes(
                dtick="M1",
                tickformat="%b\n%Y")
            fig.update_layout( yaxis_title= c)
            # fig.show()
            st_tab.plotly_chart(fig)
            print('*************************')
        
        logging.info(f"{id(self)}, completed visualize_forecast_cluster")