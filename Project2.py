import pickle
import re
import sys
import os

import pandas as pd
from sklearn import decomposition, linear_model
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import json
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from torch.utils.data import TensorDataset, DataLoader


def parse_args():
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('Usage: python Project2.py <yelp_data.jsonl> <task> <optional_load_model_name>')
        quit()
    if not os.path.isfile(sys.argv[1]) or not sys.argv[1].endswith(".jsonl"):
        print('Error: first argument is not a path to a .jsonl file')
        quit()
    if (sys.argv[2] != "Prob" and sys.argv[2] != "SVMClass" and sys.argv[2]
            != "SVRReg" and sys.argv[2] != "NNClass" and sys.argv[2] != "MLPReg"):
        print('Error: second argument must be \"Prob\" or \"SVMClass\" or \"SVRReg\" or '
              '\"NNClass\" or \"MLPReg\"')
        quit()
    if len(sys.argv) == 4 and (not os.path.isfile(sys.argv[3]) or not sys.argv[3].endswith(".pkl")):
        print('Error: fourth argument is not a path to a .pkl file')
        quit()
    if len(sys.argv) == 5 and (not sys.argv[3].endswith(".pkl") or sys.argv[4] != "cannery"):
        print('Usage: python Project2.py <yelp_data.json> <task> <optional_load_model_name>')
        quit()

    if len(sys.argv) > 3:
        pickle_input = sys.argv[3]
    else:
        pickle_input = None

    return sys.argv[1], sys.argv[2], pickle_input, len(sys.argv) == 5

class NNClassifier(nn.Module):
    def __init__(self, inputSize, layer2size, layer3size, vec):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(inputSize, layer2size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(layer2size, layer3size)
        self.fc3 = nn.Linear(layer3size, 5)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        self.vec = vec

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class MLPRegWrapper():
    def __init__(self, cool, funny, useful, vec):
        self.cool = cool
        self.funny = funny
        self.useful = useful
        self.vec = vec

class ProbWrapper():
    def __init__(self, stars, cool, funny, useful, vec):
        self.stars = stars
        self.cool = cool
        self.funny = funny
        self.useful = useful
        self.vec = vec

class SVRWrapper():
    def __init__(self, cool, funny, useful, vec):
        self.cool = cool
        self.funny = funny
        self.useful = useful
        self.vec = vec

# reads in the json file, only to the max entries and returns them as json_array,
# if max entries is set to 0 then it reads the full thing
def read_partial_json_file(filename, max_entries=0, encoding='utf-8'):
    json_array = []
    with open(filename, 'r', encoding=encoding) as file:
        if max_entries == 0:
            for line in file:
                json_array.append(json.loads(line))
        else:
            for _ in range(max_entries):
                line = file.readline()
                if not line:
                    break
                json_array.append(json.loads(line))
    return json_array

def add_missing_keys(json_array):
    for obj in json_array:
        for key in ['stars', 'useful', 'funny', 'cool', 'text']:
            if key not in obj:
                obj[key] = 0
                if key == 'stars':
                    obj[key] = 3
                print("Key {} not found in json".format(key))
    return json_array

# removes specified keys from json array
def remove_keys(json_array, keys_to_remove):
    for obj in json_array:
        for key in keys_to_remove:
            obj.pop(key, None)
    return json_array

def ConvertJSONFileToDataFrame(filename, max_entries=1000, encoding='utf-8'):
    #load in the json array
    json_array = read_partial_json_file(filename, max_entries, encoding)
    #add in the missing keys, will set to 0 for now but a heuristic for this will have to be made.
    json_array = add_missing_keys(json_array)
    df = pd.DataFrame(json_array)
    ColumnsToRemove = ['business_id', 'user_id', 'date', 'review_id']
    df = df.drop(columns=ColumnsToRemove)
    return df

# function which removes all the stop text from the 'text' column
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

def preprocess_dataframe(df, text_column):
    # Apply preprocess_text function to the specified text column in the DataFrame
    df[text_column] = df[text_column].apply(preprocess_text)
    return df

#TODO MAIN
data_path, task, pickle_path, should_pickle = parse_args()

if task == "NNClass":
    if pickle_path is not None and not should_pickle:
        print("Opening pickle...")
        in_pickle_file = open(pickle_path, "rb")
        model = pickle.load(in_pickle_file)
        in_pickle_file.close()
        print("Done.")

        dataset = ConvertJSONFileToDataFrame(data_path, 10000)

        nltk.download('stopwords')
        nltk.download('punkt')

        stem = SnowballStemmer("english")
        stopWords = stopwords.words('english')

        def stemText(text):
            return " ".join([i for i in word_tokenize(text) if not i in stopWords])

        # Data preprocessing: convert text to lowercase
        X = dataset['text'].map(lambda x: stemText(x.lower()))
        # convert star count to categories starting from 0
        translation = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        labels = ['1', '2', '3', '4', '5']
        y = dataset['stars'].copy()
        y.replace(translation, inplace=True)

        vec = model.vec
        X_vec = vec.transform(X)

        X_tensor = torch.tensor(X_vec.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y.to_numpy(), dtype=torch.long)

        outputs = model(X_tensor)
        wasteTensors, predicted = torch.max(outputs, 1)
        y_test_np = y_tensor.numpy()
        predicted_np = predicted.numpy()
        print(classification_report(y_test_np, predicted_np))
        quit()
    elif pickle_path is None or should_pickle:
        dataset = ConvertJSONFileToDataFrame(data_path, 5000)

        nltk.download('stopwords')
        nltk.download('punkt')

        stem = SnowballStemmer("english")
        stopWords = stopwords.words('english')

        def stemText(text):
            return " ".join([i for i in word_tokenize(text) if not i in stopWords])

        # Data preprocessing: convert text to lowercase
        X = dataset['text'].map(lambda x: stemText(x.lower()))
        # convert star count to categories starting from 0
        translation = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        labels = ['1', '2', '3', '4', '5']
        y = dataset['stars'].copy()
        y.replace(translation, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=117)

        vec = CountVectorizer()
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        model = NNClassifier(X_train_vec.shape[1], 128, 64, vec)

        trainDataset = TensorDataset(X_train_tensor, y_train_tensor)
        trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainingEpochs = 20
        for i in range(trainingEpochs):
            sumLoss = 0.0
            for text, stars in trainLoader:
                optimizer.zero_grad()
                result = model(text)
                loss = criterion(result, stars)
                loss.backward()
                optimizer.step()
                sumLoss += loss.item()
            print("Loss: {}".format(sumLoss))

        outputs = model(X_test_tensor)
        wasteTensors, predicted = torch.max(outputs, 1)
        y_test_np = y_test_tensor.numpy()
        predicted_np = predicted.numpy()
        print(classification_report(y_test_np, predicted_np))

        if should_pickle:
            out_file = open(pickle_path, "wb")
            pickle.dump(model, out_file)
            out_file.close()

        quit()
elif task == "MLPReg":
    if pickle_path is not None and not should_pickle:
        print("Opening pickle...")
        in_pickle_file = open(pickle_path, "rb")
        model = pickle.load(in_pickle_file)
        in_pickle_file.close()
        print("Done.")

        df = ConvertJSONFileToDataFrame(data_path, 10000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]

        X_test_regression = model.vec.transform(X)

        # Predictions
        y_pred_test_funny = model.funny.predict(X_test_regression)
        # Evaluate the model
        test_mse_funny = mean_squared_error(y['funny'], y_pred_test_funny)
        print("Funny Test MSE:", test_mse_funny)

        y_pred_test_useful = model.useful.predict(X_test_regression)

        # Evaluate the model
        test_mse_useful = mean_squared_error(y['useful'], y_pred_test_useful)
        print("Useful Test MSE:", test_mse_useful)

        # Predictions
        y_pred_test_cool = model.cool.predict(X_test_regression)

        # Evaluate the model
        test_mse_cool = mean_squared_error(y['cool'], y_pred_test_cool)

        print("Useful Test MSE:", test_mse_cool)
        quit()
    elif pickle_path is None or should_pickle:

        df = ConvertJSONFileToDataFrame(data_path, 5000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        tfid = TfidfVectorizer()
        X_train_regression = tfid.fit_transform(X_train)
        X_test_regression = tfid.transform(X_test)

        mlp_regressor_useful = MLPRegressor(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,),
                                            learning_rate='constant', learning_rate_init=0.1, solver='adam',
                                            max_iter=200, random_state=42)
        # # Train the model
        mlp_regressor_useful.fit(X_train_regression, y_train['useful'])

        # Predictions
        y_pred_test_useful = mlp_regressor_useful.predict(X_test_regression)

        # Evaluate the model
        test_mse_useful = mean_squared_error(y_test['useful'], y_pred_test_useful)
        print("Useful Test MSE:", test_mse_useful)

        mlp_regressor_funny = MLPRegressor(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,),
                                           learning_rate='constant', learning_rate_init=0.1, solver='adam',
                                           max_iter=200, random_state=42)

        # # Train the model
        mlp_regressor_funny.fit(X_train_regression, y_train['funny'])

        # Predictions
        y_pred_test_funny = mlp_regressor_funny.predict(X_test_regression)

        # Evaluate the model
        test_mse_funny = mean_squared_error(y_test['funny'], y_pred_test_funny)

        print("Funny Test MSE:", test_mse_funny)

        mlp_regressor_cool = MLPRegressor(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,),
                                          learning_rate='constant', learning_rate_init=0.1, solver='adam',
                                          max_iter=200, random_state=42)

        # # Train the model
        mlp_regressor_cool.fit(X_train_regression, y_train['cool'])

        # Predictions
        y_pred_test_cool = mlp_regressor_cool.predict(X_test_regression)

        # Evaluate the model
        test_mse_cool = mean_squared_error(y_test['cool'], y_pred_test_cool)

        print("Cool Test MSE:", test_mse_cool)

        if should_pickle:
            model = MLPRegWrapper(mlp_regressor_cool, mlp_regressor_funny, mlp_regressor_useful, tfid)
            out_file = open(pickle_path, "wb")
            pickle.dump(model, out_file)
            out_file.close()

        quit()
elif task == "Prob":
    #TODO get this to work
    if pickle_path is None or should_pickle:
        df = ConvertJSONFileToDataFrame(data_path, max_entries=5000)

        train, validation = train_test_split(df, test_size=0.2, random_state=4, shuffle=True)
        train, test = train_test_split(train, test_size=0.25, random_state=89, shuffle=True)
        df = train.groupby(by='stars').agg('count').reset_index()

        oneCount = int((df.loc[df['stars'] == 1.0]['text'].astype(int)).iloc[0])
        twoCount = int((df.loc[df['stars'] == 2.0]['text'].astype(int)).iloc[0])
        threeCount = int((df.loc[df['stars'] == 3.0]['text'].astype(int)).iloc[0])
        fourCount = int((df.loc[df['stars'] == 4.0]['text'].astype(int)).iloc[0])
        fiveCount = int((df.loc[df['stars'] == 5.0]['text'].astype(int)).iloc[0])

        # Probability distribution
        totalCount = oneCount + twoCount + threeCount + fourCount + fiveCount
        p_catagory = {'1.0': oneCount / totalCount, '2.0': twoCount / totalCount, '3.0': threeCount / totalCount,
                      '4.0': fourCount / totalCount, '5.0': fiveCount / totalCount}
        ps_catagory = [float(oneCount / totalCount), twoCount / totalCount, threeCount / totalCount,
                       fourCount / totalCount, fiveCount / totalCount]

        bigDict = [{}, {}, {}, {}, {}]
        allwords = {}

        for row_index, row in train.iterrows():
            c = int(row["stars"]) - 1
            if (c == -1):
                print("WARNING!")
            content = row['text']
            for word in content.lower().split():
                if word in bigDict[c]:
                    bigDict[c][word] += 1.0
                else:
                    bigDict[c][word] = 1.0
                if word not in allwords:
                    allwords[word] = 1.0
                else:
                    allwords[word] += 1.0

        for word in allwords.keys():
            for dict in bigDict:
                if word in dict:
                    dict[word] += 1.0
                else:
                    dict[word] = 1.0

        for dict in bigDict:
            totalCount = 0
            for word in dict:
                totalCount += dict[word]
            for word in dict:
                dict[word] = float(dict[word]) / (float(totalCount) + len(allwords))

        predictionSet = []
        # Build Train Model
        for row_index, row in test.iterrows():
            actual = row["stars"]
            content = row['text']
            p_chance = ps_catagory.copy()
            for w in content.lower().split():
                for i in range(5):
                    if w in bigDict[i]:
                        p_chance[i] = float(p_chance[i] * (bigDict[i][w]))
            pred = p_chance.index(max(p_chance))
            if (pred == 0):
                predictionSet.append(1.0)
            elif (pred == 1):
                predictionSet.append(2.0)
            elif (pred == 2):
                predictionSet.append(3.0)
            elif (pred == 3):
                predictionSet.append(4.0)
            elif (pred == 4):
                predictionSet.append(5.0)

        print(classification_report(test['stars'], predictionSet))

        lengther = len(train['cool'])
        targetVectorCool = [0] * lengther
        targetVectorUseful = [0] * lengther
        targetVectorFunny = [0] * lengther
        for i in range(lengther):
            targetVectorCool[i] = train.iloc[i]['cool']
            targetVectorUseful[i] = train.iloc[i]['useful']
            targetVectorFunny[i] = train.iloc[i]['funny']

        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), dtype='double')
        data = vectorizer.fit_transform(train['text'])
        pca = decomposition.TruncatedSVD(n_components=50)
        data = pca.fit_transform(data)
        regressionCool = linear_model.BayesianRidge()
        regressionCool.fit(data, targetVectorCool)
        predictCool = regressionCool.predict(pca.fit_transform(vectorizer.fit_transform(test['text'])))
        mean_squared_error(test['cool'], predictCool)
        regressionUseful = linear_model.BayesianRidge()
        regressionUseful.fit(data, targetVectorUseful)
        predictUseful = regressionUseful.predict(pca.fit_transform(vectorizer.fit_transform(test['text'])))
        mean_squared_error(test['useful'], predictUseful)
        regressionFunny = linear_model.BayesianRidge()
        regressionFunny.fit(data, targetVectorFunny)
        predictFunny = regressionFunny.predict(pca.fit_transform(vectorizer.fit_transform(test['text'])))
        mean_squared_error(test['funny'], predictFunny)

        if should_pickle:
            model = (ps_catagory, bigDict)
            out_file = open(pickle_path, "wb")
            pickle.dump(model, out_file)
            out_file.close()
        quit()
    elif not should_pickle and pickle_path is not None:
        print("Opening pickle...")
        in_pickle_file = open(pickle_path, "rb")
        model = pickle.load(in_pickle_file)
        in_pickle_file.close()
        print("Done.")
        df = ConvertJSONFileToDataFrame(data_path, 10000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]

        ps_catagory, bigDict = model

        predictionSet = []
        # Build Train Model
        for row in range(len(X)):
            content = X.iloc[row]
            p_chance = ps_catagory.copy()
            for w in content.lower().split():
                for i in range(5):
                    if w in bigDict[i]:
                        p_chance[i] = float(p_chance[i] * (bigDict[i][w]))
            pred = p_chance.index(max(p_chance))
            if (pred == 0):
                predictionSet.append(1.0)
            elif (pred == 1):
                predictionSet.append(2.0)
            elif (pred == 2):
                predictionSet.append(3.0)
            elif (pred == 3):
                predictionSet.append(4.0)
            elif (pred == 4):
                predictionSet.append(5.0)
        print(classification_report(y['stars'], predictionSet))
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 1), dtype='double')
        data = vectorizer.fit_transform(X)
        pca = decomposition.TruncatedSVD(n_components=50)
        data = pca.fit_transform(data)
        regressionCool = linear_model.BayesianRidge()
        regressionCool.fit(data, targetVectorCool)
        predictCool = regressionCool.predict(pca.fit_transform(vectorizer.fit_transform(X)))
        mean_squared_error(y['cool'], predictCool)
        regressionUseful = linear_model.BayesianRidge()
        regressionUseful.fit(data, targetVectorUseful)
        predictUseful = regressionUseful.predict(pca.fit_transform(vectorizer.fit_transform(X)))
        mean_squared_error(y['useful'], predictUseful)
        regressionFunny = linear_model.BayesianRidge()
        regressionFunny.fit(data, targetVectorFunny)
        predictFunny = regressionFunny.predict(pca.fit_transform(vectorizer.fit_transform(X)))
        mean_squared_error(y['funny'], predictFunny)
        quit()
elif task == "SVMClass":
    if pickle_path is None or should_pickle:
        df = ConvertJSONFileToDataFrame(data_path, 1000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        svm = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', SVC(kernel='linear', C=1))
        ])
        # Train the model
        svm.fit(X_train, y_train['stars'])

        y_pred = svm.predict(X_test)
        label_report_stars = classification_report(y_test['stars'], y_pred, zero_division=1)
        print("Classification Report for label", label_report_stars)

        if should_pickle:
            out_file = open(pickle_path, "wb")
            pickle.dump(svm, out_file)
            out_file.close()
        quit()
    elif pickle_path is not None and not should_pickle:
        print("Opening pickle...")
        in_pickle_file = open(pickle_path, "rb")
        svm = pickle.load(in_pickle_file)
        in_pickle_file.close()
        print("Done.")

        df = ConvertJSONFileToDataFrame(data_path, 10000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]

        y_pred = svm.predict(X)

        label_report_stars = classification_report(y['stars'], y_pred, zero_division=1)
        print("Classification Report for label", label_report_stars)
        quit()
elif task == "SVRReg":
    if pickle_path is None or should_pickle:
        df = ConvertJSONFileToDataFrame(data_path, 1000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        tfid = TfidfVectorizer()
        X_train_regression = tfid.fit_transform(X_train)
        X_test_regression = tfid.transform(X_test)

        # Create SVR model for cool
        svr_model_cool = SVR(kernel='rbf', gamma='scale', epsilon=0.1,
                             C=1)  # Radial Basis Function (RBF) kernel is commonly used
        # Other kernels you can try: 'linear', 'poly', 'sigmoid'

        # Train the model
        svr_model_cool.fit(X_train_regression, y_train['cool'])

        # Predict on the test set
        y_pred = svr_model_cool.predict(X_test_regression)

        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y_test['cool'], y_pred)
        print("Cool Mean Squared Error:", mse)

        # Create SVR model for funny
        svr_model_f = SVR(kernel='rbf', gamma='scale', epsilon=0.1, C=1)

        # Train the model
        svr_model_f.fit(X_train_regression, y_train['funny'])

        # Predict on the test set
        y_pred = svr_model_f.predict(X_test_regression)

        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y_test['funny'], y_pred)
        print("Funny Mean Squared Error:", mse)
        # Create SVR model for useful
        svr_model_u = SVR(kernel='rbf', gamma='scale', epsilon=0.1,
                          C=1)  # Radial Basis Function (RBF) kernel is commonly used
        # Other kernels you can try: 'linear', 'poly', 'sigmoid'

        # Train the model
        svr_model_u.fit(X_train_regression, y_train['useful'])

        # Predict on the test set
        y_pred = svr_model_u.predict(X_test_regression)

        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y_test['useful'], y_pred)
        print("Useful Mean Squared Error:", mse)

        if should_pickle:
            model = SVRWrapper(svr_model_cool, svr_model_f, svr_model_u, tfid)
            out_file = open(pickle_path, "wb")
            pickle.dump(model, out_file)
            out_file.close()

        quit()
    elif pickle_path is not None and not should_pickle:
        print("Opening pickle...")
        in_pickle_file = open(pickle_path, "rb")
        model = pickle.load(in_pickle_file)
        in_pickle_file.close()
        print("Done.")

        df = ConvertJSONFileToDataFrame(data_path, 10000)
        df = preprocess_dataframe(df, 'text')

        target_columns = ['stars', 'cool', 'funny', 'useful']

        X = df['text']
        y = df[target_columns]

        X_test_regression = model.vec.transform(X)

        y_pred = model.useful.predict(X_test_regression)
        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y['useful'], y_pred)
        print("Useful Mean Squared Error:", mse)

        y_pred = model.funny.predict(X_test_regression)
        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y['funny'], y_pred)
        print("Funny Mean Squared Error:", mse)

        y_pred = model.cool.predict(X_test_regression)
        # Calculate Mean Squared Error (MSE) as a metric
        mse = mean_squared_error(y['cool'], y_pred)
        print("Cool Mean Squared Error:", mse)

