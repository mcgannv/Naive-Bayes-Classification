# Import LabelEncoder
from sklearn import preprocessing
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB


weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Overcast', 'Snowy', 'Snowy', 'Overcast', 'Sunny',
           'Sunny', 'Overcast', 'Overcast', 'Rainy', 'Sunny']
temp = ['Hot', 'Hot', 'Mild', 'Cold', 'Mild', 'Mild', 'Cold', 'Cold', 'Mild', 'Mild', 'Mild', 'Cold', 'Hot', 'Mild',
        'Hot', 'Mild']
yard_work = ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
netflix = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
sports = ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']

# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
# print(weather_encoded)

temp_encoded = le.fit_transform(temp)
# print(temp_encoded)

yard_work_label = le.fit_transform(yard_work)
# print(yard_work_label)

netflix_label = le.fit_transform(netflix)
# print(netflix_label)

sports_label = le.fit_transform(sports)
# print(sports_label)

features = list(zip(weather_encoded, temp_encoded))
print(features)

# Create a Gaussian Classifier
netflix_model = GaussianNB()
yard_model = GaussianNB()
sports_model = GaussianNB()

# Train the model using the training sets
netflix_model.fit(features,netflix_label)
yard_model.fit(features,yard_work_label)
sports_model.fit(features,sports_label)

# Weather Sunny: 3, Overcast: 0, Rainy: 1, Snowy: 2
# Temp Hot: 1, Mild: 2, Cold: 0
predict_data = [2, 2]

# Predict Output
netflix_predicted = netflix_model.predict([predict_data])
yard_predicted = yard_model.predict([predict_data])
sports_predicted = sports_model.predict([predict_data])
print("Yard Work: ", yard_predicted)
print("Netflix: ", netflix_predicted)
print("Sports: ", sports_predicted)
print("Testinggggg")
