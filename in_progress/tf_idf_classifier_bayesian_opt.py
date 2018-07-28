import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from utils import DataProcessor
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
# twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


space  = [Integer(1, 4, name='ngram_range'),
          Real(0.01, 10**0,  name='max_df'),
          Integer(1, 5000, name='max_features'),
        #   Real(0.01, 1000,"log-uniform", name='alpha')
          Categorical(['squared_hinge','hinge'],name='loss'),
          Real(0.01, 1000,"log-uniform", name='C')]



class Model(object):
    def __init__(self,data_file):
        data_processor = DataProcessor(data_file,seperator=',,,')
        self.data , self.labels    = data_processor.get_training_data(raw_text=True)
        # self.data , self.labels    = twenty_train.data, twenty_train.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                self.data, self.labels, test_size=0.33, random_state=42)

        # print('Running Naive Bayes...')
        # pipeline, parameters =self.get_naive_bayes_model()

        @use_named_args(space)
        def objective(**params):
            print params
            # max_df,ngram_range,max_features,alpha
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_df=params['max_df'],ngram_range=(1,params['ngram_range']),\
                                max_features=params['max_features'])),
                ('clf', LinearSVC(loss=params['loss'],C=params['C'], max_iter=1000))
            ])
            pipeline.fit(self.X_train,self.y_train)
            accuracy = accuracy_score(pipeline.predict(self.X_test),self.y_test)
            print('Accuracy {}'.format(accuracy))
            return -accuracy

        res_gp = gp_minimize(objective, space, n_calls=100, random_state=10)


    #     grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=0)
    #     grid_search_tune.fit(self.X_train, self.y_train)
    #     print("Best parameters set:")
    #     self.best_estimator_ =  grid_search_tune.best_estimator_
    #     print(grid_search_tune.best_score_)
    #     self.calculate_metric()
    #     print('#'*80)
    #
    #     print('Running Linear SVM...')
    #     pipeline, parameters = self.get_linear_svm_model()
    #     grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=0)
    #     grid_search_tune.fit(self.X_train, self.y_train)
    #     print("Best parameters set:")
    #     self.best_estimator_ =  grid_search_tune.best_estimator_
    #     print(grid_search_tune.best_score_)
    #     self.calculate_metric()
    #     print('#'*80)
    #
    #     print('Running Non Linear SVM...')
    #     pipeline, parameters = self.get_non_linear_svm_model()
    #     grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=0)
    #     grid_search_tune.fit(self.X_train, self.y_train)
    #     print("Best parameters set:")
    #     self.best_estimator_ =  grid_search_tune.best_estimator_
    #     print(grid_search_tune.best_score_)
    #     self.calculate_metric()
    #     print('#'*80)
    #
    # def get_naive_bayes_model(self):
    #     pipeline = Pipeline([
    #         ('tfidf', TfidfVectorizer()),
    #         ('clf', MultinomialNB(fit_prior=True))
    #     ])
    #     parameters = {
    #         'tfidf__max_df': (0.25, 0.5, 0.75),
    #         'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #         'tfidf__max_features': (100,500,1000,5000, 10000),
    #         'clf__alpha':(1e-3,1e-2,1e-1,1)
    #     }
    #
    #     return pipeline, parameters



    def get_linear_svm_model(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC( max_iter=1000))
        ])

        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__max_features': (100,500,1000,5000, 10000),
            # 'clf__penalty':('l1','l2'),
            'clf__loss':('squared_hinge','hinge'),
            'clf__C':(1,5,10),
        }
        return pipeline, parameters

    def get_non_linear_svm_model(self):


        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC())
        ])

        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__max_features': (100,500,1000,5000, 10000),
            'clf__kernel':('rbf','sigmoid'),
            'clf__C':(1,5,10),
        }
        return pipeline, parameters

    def calculate_metric(self):
        print('Test Metrics:')
        predicted = self.best_estimator_.predict(self.X_test)
        print('Accuracy:', np.mean(predicted == self.y_test))
        print(classification_report(self.y_test,predicted))

    def calculate_accuray(self,y_true,y_pred):
        return accuracy_score(y_true,y_pred)
        # print('Test Score:', np.mean(predicted == self.y_test))

if __name__ == '__main__':
    data_path = 'data/custom/LabelledData.txt'
    temp = Model(data_path)
