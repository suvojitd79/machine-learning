'''

    given a list of brain-body weights. Predict the body weight for a  
    given brain value. y = mx + c 

'''

import numpy as np 
from sklearn import linear_model
import pandas as pd 
import matplotlib.pyplot as plot 

#data-set
data = pd.read_fwf('data.txt')
x = data[['Brain']]
y = data[['Body']]

#train the model
model = linear_model.LinearRegression()
model.fit(x,y)

#visualize the data
plot.scatter(x,y,color='red')
plot.plot(x,model.predict(x))
plot.show()



#make some predictions......
test = pd.DataFrame([[192]],columns = ['Brain'])
print(model.predict(test))