import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import numpy as np
%matplotlib inline

""" Columnas del df properati

Index(['id', 'ad_type', 'start_date', 'end_date', 'created_on', 'lat', 'lon',
       'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'rooms', 'bedrooms', 'bathrooms',
       'surface_total', 'surface_covered', 'price', 'currency', 'price_period',
       'title', 'description', 'property_type', 'operation_type'],
      dtype='object')
"""

inmueble = "https://www.dropbox.com/s/mswj6wt0ymxlcbe/ar_properties.csv"
df = pd.read_csv(inmueble, error_bad_lines=False)


df.info()
df.head()
df2 = df.head(30)
df.columns


venta = df[df["operation_type"]=="Venta"]
venta["currency"].value_counts()
usd = venta[venta["currency"]=="USD"]
usd["property_type"].value_counts()
deptos = usd[usd["property_type"]=="Departamento"]
deptos["bedrooms"].value_counts()

cdf0 = deptos[['l2', 'l3', 'rooms', 'bedrooms', 'bathrooms', 'surface_total', 'surface_covered', 'price', 'property_type']]
cdf = cdf0.dropna()


cdf =cdf[cdf["surface_total"]<300]
cdf =cdf[cdf["price"]<5000000]

sns.heatmap(cdf.corr())

plt.scatter(cdf.surface_total, cdf.price,  color='blue')
plt.xlabel("sup. total")
plt.ylabel("valor")
plt.show()


msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]


plt.scatter(train.surface_total, train.price,  color='green')
plt.xlabel("surface_total")
plt.ylabel("price")
plt.show()


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['surface_total',"rooms", 'bedrooms', 'surface_total']])
y = np.asanyarray(train[['price']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

"""
As mentioned before, Coefficient and Intercept in the simple linear regression, 
are the parameters of the fit line. Given that it is a simple linear regression, 
with only 2 parameters, and knowing that the parameters are the intercept and slope of the line,
 sklearn can estimate them directly from our data. Notice that all of the data must be available
 to traverse and calculate the parameters.
"""

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

y_= regr.predict(test[['surface_total',"rooms", 'bedrooms', 'surface_total']])
x = np.asanyarray(test[['surface_total',"rooms", 'bedrooms', 'surface_total']])
y = np.asanyarray(test[['price']])
print("Residual sum of squares: %.5f"% np.mean((y_ - y) ** 2))
print('Variance score: %.5f' % regr.score(x, y))

#we can plot the fit line over the data:
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


"""
Evaluation
we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model, as it provides insight to areas that require improvement.

There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

Mean absolute error: It is the mean of the absolute value of the errors. This is the easiest of the metrics to understand since it’s just average error.
Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially increasing larger errors in comparison to smaller ones.
Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.
R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

"""






#df.dropna(how="any",inplace=True)
