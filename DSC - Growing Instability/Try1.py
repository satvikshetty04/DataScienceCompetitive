import json
import collections
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
import sys


##### Reading the Json file for Training set #####
print("Reading the input data...")
files = ['1999a_TrainingData.json','1999b_TrainingData.json','2000a_TrainingData.json']

##### Reading Json file for Test data       #####
with open('2000b_TrainingData.json', 'r') as content_file:
    json_data = content_file.read()
py_object_test_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TrainingData']
test_data_x = []
test_data_y = []
for key, value in py_object_test_data.items():
    test_data_x.append(value["bodyText"])
    test_data_y.append(value["topics"])

##### Cleaning the Test data, filtering the missing values       #####
test_data_frame = pd.DataFrame(
    {'bodyText': test_data_x,
     'topics': test_data_y,
     })
test_data_frame = test_data_frame[test_data_frame.topics.apply(lambda c: c != [])]
test_data_frame = test_data_frame[test_data_frame.bodyText.apply(lambda c: c != "")]
test_data_x = test_data_frame.bodyText
test_data_y = test_data_frame.topics

vectorizer = TfidfVectorizer(min_df=0.001, max_df=0.90, stop_words='english', smooth_idf=True,
                             norm="l2", sublinear_tf=False, use_idf=True, ngram_range=(1, 3))


classifier = OneVsRestClassifier(SGDClassifier(alpha=0.00001))

##### Level binarizer for Class variable of Test set     #####
mlb_test = MultiLabelBinarizer()
test_data_y = mlb_test.fit_transform(test_data_y)

for file in files:
    print("Opening ",file)
    with open(file, 'r') as content_file:
        json_data = content_file.read()
    py_object_train_data = json.loads(json_data, object_pairs_hook=collections.OrderedDict)['TrainingData']
    train_data_x = []
    train_data_y = []
    for key, value in py_object_train_data.items():
        train_data_x.append(value["bodyText"])
        train_data_y.append(value["topics"])

    ##### Cleaning the Train data, filtering the missing values       #####
    train_data_frame = pd.DataFrame(
        {'bodyText': train_data_x,
         'topics': train_data_y,
        })
    train_data_frame = train_data_frame[train_data_frame.topics.apply(lambda c: c != [])]
    train_data_frame = train_data_frame[train_data_frame.bodyText.apply(lambda c: c != "")]
    train_data_x = train_data_frame.bodyText
    train_data_y = train_data_frame.topics

    print("Generating frequency vectors for the words present in the training set. This will take a while...")
    X = vectorizer.fit_transform(train_data_x)
    print(vectorizer)

    ##### Generate vectors for words in test data Articles    #####
    print(vectorizer)
    xTest = vectorizer.transform(test_data_x)

    ##### Level binarizer for Class variable of Train set     #####
    print("Converting the class variable to binary matrix...")
    mlb_train = MultiLabelBinarizer()
    train_data_y = mlb_train.fit_transform(train_data_y)

    print("*******")
    print(train_data_y)
    print("*******")
    print(classifier)
    print("*******")
    ##### Fitting a One Vs Rest Classifier using the Naive Bayes classifier       #####
    print("Training the model...")

    classifier = classifier.fit(X, train_data_y)

    print(classifier)
    sys.exit(1)
    ##### Getting the Predictions for labels for test data        #####
    print("Getting the predictions for labels on Test Data...")
    y_pred = classifier.predict(xTest)
    print(y_pred)
    ##### Getting predicted labels and Actual labels present in test data for all test samples    #####
    dict_test_y = {}
    dict_test_pred = {}
    for i in range(len(test_data_y)):
        list_pred_y = []
        for j in range(len(list(mlb_train.classes_))):
            if (y_pred[i][j] == 1):
                list_pred_y.append(j)
        dict_test_pred[i] = list_pred_y

    for i in range(len(test_data_y)):
        list_test_y = []
        for j in range(len(list(mlb_test.classes_))):
            if (test_data_y[i][j] == 1):
                list_test_y.append(j)
        dict_test_y[i] = list_test_y

    ##### Calculating the accuracy measures       #####
    print("Checking for Precision and F-Measure...")
    correct_pred = 0
    total_labels = 0
    total_pred = 0
    for i in dict_test_y.keys():
        predicted = []
        tested = []
        for k in dict_test_pred[i]:
            predicted.append(list(mlb_train.classes_)[k])
        for k in dict_test_y[i]:
            tested.append(list(mlb_test.classes_)[k])
        total_labels = total_labels + len(tested)
        total_pred = total_pred + len(predicted)
        correct_pred = correct_pred + len(set(predicted).intersection(tested))

    print("\nTotal labels predicted: " + str(total_pred))
    print("Total labels in Test Data: " + str(total_labels))
    print("Correct predicted labels: " + str(correct_pred))

    precision = float(correct_pred * 100 / total_pred)
    recall = float(correct_pred * 100 / total_labels)
    f_measure = float(((2 * correct_pred * 100) / (total_labels + total_pred)))

    print("\nPrecision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print("F-Measure: %.2f" % f_measure)

# def main():
#     ##### Calling the Naive Bayes Classifier #####
#     #print("\nNAIVE BAYES CLASSIFIER...\n")
#     #NB(train_data_x, train_data_y, test_data_x, test_data_y)
#     ##### Calling the SGD Classifier #####
#     print("\nSGD CLASSIFIER...\n")
#     SGD(train_data_x, train_data_y, test_data_x, test_data_y)
#
# if __name__ == "__main__":
#     main()