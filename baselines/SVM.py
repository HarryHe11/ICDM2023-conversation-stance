from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
# assume X_train and y_train are the training data
# assume X_test is the test data
data_path = r'./data'
train_path = data_path + '/train.txt'
test_path = data_path + '/test.txt'
X_train = []
y_train = []
X_test = []
y_test = []
def read_file(data_path):
    contents = []
    labels = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            contents.append(content)
            labels.append(int(label))
    return contents,labels

X_train, y_train = read_file(train_path)
X_test, y_test = read_file(test_path)
# create a CountVectorizer object with n-gram range (1, 3)
vectorizer = CountVectorizer(ngram_range=(1, 3))

# fit the vectorizer on the training data and transform it into feature vectors
X_train_vec = vectorizer.fit_transform(X_train)

# transform the test data into feature vectors using the same vectorizer
X_test_vec = vectorizer.transform(X_test)

# create an SVM classifier with linear kernel and C=1.0
clf = SVC(kernel='linear', C=1.0)

# train the classifier on the training data
clf.fit(X_train_vec, y_train)

# predict labels for the test data using the trained classifier
y_pred = clf.predict(X_test_vec)

print(accuracy_score(y_test,y_pred))
print(f1_score(y_test,y_pred, average='macro'))
print(classification_report(y_test,y_pred))
