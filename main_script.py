import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import graphviz
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3

# Keras specific
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

st.image("https://www.isb.edu/content/dam/sites/isb/about-isb/ISB_transparent_logo.png", width=400)

st.title('Machine Learning Predictor')

# Main Predicor class
class Predictor:
    # Data preparation part, it will automatically handle with your data
    def prepare_data(self, split_data, train_test):
        # Reduce data size
        data = self.data[self.features]
        data = data.sample(frac = round(split_data/100,2))

        # Impute nans with mean for numeris and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
            data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
        imp = SimpleImputer(missing_values = np.nan, strategy="mean")
        data.loc[:,data.dtypes != 'object'] = imp.fit_transform(data.loc[:,data.dtypes != 'object'])

        # One hot encoding for categorical variables
        cats = data.dtypes == 'object'
        le = LabelEncoder() 
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:,x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder() 
        data.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(data.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names()))

        # Set target column
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)  

    # Classifier type and algorithm selection 
    def set_classifier_properties(self):
        self.type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression", "Clustering"))
        if self.type == "Regression":
            self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Random Forest', 'Linear Regression', 'Neural Network')) 
            if self.chosen_classifier == 'Random Forest': 
                self.n_trees = st.sidebar.slider('number of trees', 1, 1000, 1)
            elif self.chosen_classifier == 'Neural Network':
                self.epochs = st.sidebar.slider('number of epochs', 1 ,100 ,10)
                self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
        elif self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Logistic Regression', 'Naive Bayes', 'Neural Network')) 
            if self.chosen_classifier == 'Logistic Regression': 
                self.max_iter = st.sidebar.slider('max iterations', 1, 100, 10)
            elif self.chosen_classifier == 'Neural Network':
                self.epochs = st.sidebar.slider('number of epochs', 1 ,100 ,10)
                self.learning_rate = float(st.sidebar.text_input('learning rate:', '0.001'))
                self.number_of_classes = int(st.sidebar.text_input('Number of classes', '2'))

        
        elif self.type == "Clustering":
            pass

    # Model training and predicitons 
    def predict(self, predict_btn):    

        if self.type == "Regression":    
            if self.chosen_classifier == 'Random Forest':
                self.alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=self.n_trees)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
                
            
            elif self.chosen_classifier=='Linear Regression':
                self.alg = LinearRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier=='Neural Network':
                model = Sequential()
                model.add(Dense(500, input_dim = len(self.X_train.columns), activation='relu',))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))

                # optimizer = keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss= "mean_squared_error" , optimizer='adam', metrics=["mean_squared_error"])
                self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=40)
                self.predictions = model.predict(self.X_test)
                self.predictions_train = model.predict(self.X_train)

        elif self.type == "Classification":
            if self.chosen_classifier == 'Logistic Regression':
                self.alg = LogisticRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions
        
            elif self.chosen_classifier=='Naive Bayes':
                self.alg = GaussianNB()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier=='Neural Network':
                model = Sequential()
                model.add(Dense(500, input_dim = len(self.X_train.columns), activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(self.number_of_classes, activation='softmax'))

                optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=40)

                self.predictions = model.predict_classes(self.X_test)
                self.predictions_train = model.predict_classes(self.X_train)

           

        result = pd.DataFrame(columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.predictions
        result_train['Prediction_Train'] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if self.type == 'Regression':
            self.error_metrics['MSE_test'] = mean_squared_error(self.y_test, self.predictions)
            self.error_metrics['MSE_train'] = mean_squared_error(self.y_train, self.predictions_train)
            return st.markdown('### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) + 
            ' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3)))

        elif self.type == 'Classification':
            self.error_metrics['Accuracy_test'] = accuracy_score(self.y_test, self.predictions)
            self.error_metrics['Accuracy_train'] = accuracy_score(self.y_train, self.predictions_train)
            return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
            ' -- Accuracy Test: ' + str(round(self.error_metrics['Accuracy_test'], 3)))

    # Plot the predicted values and real values
    def plot_result(self):
        
        output_file("slider.html")

        s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
        s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color="Black", alpha=1, legend_label = "Actual")
        s1.triangle(self.result_train.index, self.result_train.Prediction_Train, size=12, color="Red", alpha=1, legend_label = "Prediction")
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
            s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1, legend_label = "Actual")
            s2.triangle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1, legend_label = "Prediction")
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[ tab1, tab2 ])
        else:

            tabs = Tabs(tabs=[ tab1])

        st.bokeh_chart(tabs)

       
    # File selector module for web app
    def file_selector(self):
        file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            return data
        else:
            st.text("Please upload a csv file")
        
    
    def print_table(self):
        if len(self.result) > 0:
            result = self.result[['Actual', 'Prediction']]
            st.dataframe(result.sort_values(by='Actual',ascending=False).style.highlight_max(axis=0))
    
    def set_features(self):
        self.features = st.multiselect('Please choose the features including target variable that go into the model', self.data.columns )

if __name__ == '__main__':
    controller = Predictor()
    
    try:
        controller.data = controller.file_selector()

        if controller.data is not None:
            split_data = st.sidebar.slider('Randomly reduce data size %', 1, 100, 10 )
            train_test = st.sidebar.slider('Train-test split %', 1, 99, 66 )
        controller.set_features()
        if len(controller.features) > 1:
            controller.prepare_data(split_data, train_test)
            controller.set_classifier_properties()
            predict_btn = st.sidebar.button('Predict')  
    except (AttributeError, ParserError, KeyError) as e:
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>', unsafe_allow_html=True)  


    if controller.data is not None and len(controller.features) > 1:

        if predict_btn:
            st.sidebar.text("Progress:")
            my_bar = st.sidebar.progress(0)
            predictions, predictions_train, result, result_train = controller.predict(predict_btn)
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)

            controller.plot_result()
            controller.get_metrics()        
            controller.print_table()

            data = controller.result.to_csv(index=False)
            b64 = base64.b64encode(data.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.sidebar.markdown(href, unsafe_allow_html=True)


    
    if controller.data is not None:
        if st.sidebar.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(controller.data)
    






