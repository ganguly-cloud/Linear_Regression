### -*- coding: utf-8 -*-
##import numpy as np
##import pandas as pd
##import numpy as np
##import pandas as pd
##import matplotlib.pyplot as plt
##import seaborn as sns
##
##
##x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
##y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
### number of observations/points 
##n = np.size(x)
###Lets plot a scatter plot for the given values 
##colors = np.random.rand(n)
##area = 25 # 0 to 15 point radii
##plt.scatter(x, y, area, colors, alpha=0.5)
###plt.savefig('before_pred')
###plt.show()
##'''
##y= bo + b1X     # or y =mx+c
###bo  or c is intercept
###b1  or m is slope
##b0 or c= (Σy)(Σx2) - (Σx)(Σxy)/ n(Σx2) - (Σx)2
##b1 or m=(slope)= n (Σxy) - (Σx)(Σy) /n(Σx2) - (Σx)2  '''
##
###mean of x and y vector
##
##mean_x, mean_y = np.mean(x), np.mean(y)
##
### calculating cross-deviation and deviation about x 
##SS_xy = np.sum(y*x) - n*mean_y*mean_x 
##SS_xx = np.sum(x*x) - n*mean_x*mean_x 
##  
### calculating regression coefficients 
##b1 = SS_xy / SS_xx 
##b0 = mean_y - b1*mean_x 
##  
##print "Coefficient b1 is: ",b1 
##print "Coefficient b0 is: ",b0 
##'''
##Coefficient b1 is:  1.1696969696969697
##Coefficient b0 is:  1.2363636363636363'''
##
###Step 3 : Let's plot the scatter plot along with predicted y value #based on our slope & intercept
###plotting the actual points as scatter plot 
##plt.scatter(x, y, color = "m", marker = "o", s = 100) 
### predicted response vector 
##y_pred = b0 + b1*x
### plotting the regression line 
##plt.plot(x, y_pred, color = "g") 
##  
### putting labels 
##plt.xlabel('x') 
##plt.ylabel('y') 
##  
###show plot
###plt.savefig('After_reg_line')
###plt.show()
##
### using SKLEARN
##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1) 
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
#invoke the LinearRegression function and find the bestfit model on #our given data
regression_model = LinearRegression()
regression_model.fit(x, y) #this will give the best fit line 
# Let us explore the coefficients for each of the independent #attributes
b1 = regression_model.coef_
b0 = regression_model.intercept_
print("b1 is: {} and b0 is: {}".format(b1, b0))
#y_pred is the predicted value which our linear regression model
#predicted when we plotted the best fit line
y_pred = regression_model.predict(x)
print y_pred


plt.scatter(x, y, color = "m", marker = "o", s = 100) 
plt.plot(x, b1*x+b0)  or  plt.plot(x, y_pred)
plt.show()

from sklearn.metrics import r2_score

r2Score = r2_score(y, y_pred) #here y is our original value 
print r2Score  # 0.952538038613988



import pandas as pd

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print data.head()
