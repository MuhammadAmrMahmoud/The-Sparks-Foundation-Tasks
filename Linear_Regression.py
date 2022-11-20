import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

# Plotting the distribution of scores
df.plot.scatter(x='Hours', y='Scores', title='Graph of hours and scores percetages')
plt.show()

# Adding a dimension to avoid an error
y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training complete.")

line = regressor.coef_ * X + regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()

line_Prediction = regressor.predict([[9.25]])
print('If a student studies for 9.25 hours he will get : ' , line_Prediction , '%')

y_pred = regressor.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

from sklearn.metrics import mean_absolute_error  
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred)) 
