import pandas as pd
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app import app

cv = CountVectorizer()


@app.route("/train")
def train():
    df = pd.read_csv("./SpamCollection.csv", encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label_encoded'] = df.label.map({"ham": 0, "spam": 1})
    y = df['label_encoded']
    x = df['message']

    x = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=111)

    classifier = MultinomialNB(alpha=0.25)
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE value: "+ str(mse))
    joblib.dump(classifier, 'SpamClassificationModel.pkl')

    return jsonify({"status": "Model successfully trained.", "classification_report": report, "mse": mse})
#
#
# @app.route("/")
# @app.route("/index")
# def home():
#     return render_template('home.html')
#

@app.route('/predict', methods=['POST'])
def predict():
    Model = open("SpamClassificationModel.pkl", 'rb')
    classifier = joblib.load(Model)

    filepath = open("vocab.pkl", 'rb')
    vocabulary = joblib.load(filepath)
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=1, vocabulary=vocabulary)
    vectorizer._validate_vocabulary()

    message = request.get_json( )
    #message = request.form.get('message')
    sms = message["message"]
    data = [sms]
    transformed = vectorizer.transform(data).toarray()
    prediction = classifier.predict(transformed)
    result=""
    if prediction[0]==1:
        result="SPAM"
    else:
        result="HAM"
    return jsonify({"prediction": result})
