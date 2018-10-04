import pandas as pd 
import seaborn as sb 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, cross_val_score 

iris_data_clean = pd.read_csv('iris-data-clean.csv')
 
all_inputs = iris_data_clean[['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm']].values

all_labels = iris_data_clean['class'].values

random_forest_classifier = RandomForestClassifier(criterion='gini', max_features=3, n_estimators=50) 

rf_classifier_scores = cross_val_score(random_forest_classifier, all_inputs, all_labels, cv=10) 


#shows from of the predicitions 
(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25) 
random_forest_classifier.fit(training_inputs, training_classes) 

for input_features, prediction, actual in zip(testing_inputs[:10], random_forest_classifier.predict(testing_inputs[:10]), testing_classes[:10]):    print('{}\t-->\t{}\t(Actual: {})'.format(input_features, prediction, actual))
