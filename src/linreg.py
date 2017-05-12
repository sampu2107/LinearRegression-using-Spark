# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
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
    l[0] = 1.0 
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

  sc = SparkContext(appName="LinearRegression")

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

  #Calculate inverse of A mutiplied by B
  beta = np.dot(np.linalg.inv(A),B)

  #Convert the matrix to list for displaying
  beta = np.array(beta).tolist()
  
  # print the linear regression coefficients in desired output format
  print "beta coefficients: "
  for coeff in beta:
      print coeff[0]

  sc.stop()
