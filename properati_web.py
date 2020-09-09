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


