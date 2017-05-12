# linreg_GradientDescent_Convergence.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by gradient descent approach for the estimate of beta.
# 
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg_GradientDescent_Convergence.py <inputdatafile>
# Example usage: spark-submit linreg_GradientDescent_Convergence.py yxlin.csv
#
#

'''
|       Assignment:  Individual assignment: Programming - 4
|       Author:  Sampath Kumar Gunasekaran(sgunase2@uncc.edu)
|       Grader:  Walid Shalaby
|
|       Course:  ITCS 6190
|       Instructor:  Srinivas Akella
|       Due Date:  April 16 at 11:59PM
|
|       Language:  Python 
|       
|                
|       Deficiencies:  No logical errors.
'''

import sys
import numpy as np

from pyspark import SparkContext

#Function to find X * (X Transpose)
def AComponent(l):
    l[0]=1.0
    x = np.array(l).astype('float')
    X = np.asmatrix(x).T
    XXT = np.dot(X,X.T)
    return XXT

#Function to find X * Y
def BComponent(l):
    Y = float(l[0]) #Y value
    l[0] = 1.0 
    x = np.array(l).astype('float')
    X = np.asmatrix(x).T
    XY = np.multiply(X,Y)
    return XY


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegressionGradientDescent")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  #Calculate A
  A = np.asmatrix(yxlines.map(lambda l: ("KeyA",AComponent(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])

  #Calculate B
  B = np.asmatrix(yxlines.map(lambda l: ("KeyB",BComponent(l))).reduceByKey(lambda x1,x2: np.add(x1,x2)).map(lambda l: l[1]).collect()[0])

  # Calculate Beta using convergence: Beta = Beta + alpha*(B-(A*Beta))  
  alpha = 0.01 #Choosing the step size as 0.01
  beta_old = np.zeros(yxlength, dtype=float)
  converged = False
  iterations = 0
  while not converged: #Checking for convergence
	beta = np.add(beta,np.dot(alpha,np.subtract(B,np.dot(A,beta))))
        if iterations == 500:
		beta = np.array([0, 0])
		beta_old = np.array([0, 0])
		alpha = alpha/10 #Cutting down the step size
		iterations = 0
	elif np.isnan(beta).any() or np.isinf(beta).any():
		beta = np.array([0, 0])
		beta_old = np.array([0, 0])
		alpha = alpha/10 #Cutting down the step size
                iterations = 0
	elif not np.array_equal(beta,beta_old):
		beta_old = beta
		iterations += 1
	else:
		converged = True
  
  #Convert the matrix to list for displaying
  beta = np.array(beta).tolist()
  
  # print the linear regression coefficients in desired output format
  print "beta coefficients: "
  for coeff in beta:
      print coeff[0]

  sc.stop()
