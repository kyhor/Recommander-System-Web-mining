# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 07:42:12 2017

@author: Khor_000
"""

import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def Q1():
    print "Reading data..."
    data = list(parseData("/C://Users/Khor_000/Desktop/Recommander Sys&Web mining/beer_50000.json"))
    print "done"
    
    print 'Question 1:'
    X = [d['review/taste'] for d in data]
    sum1 = 0
    max1 = 0
    
    for x in X:
        sum1 += x
        if max1 < x:
            max1 = x
    
    mean = sum1/len(X)
    print 'max1', max1
    print 'mean: ', mean
    sum2  = float(0)
    
    for x in X:
        sum2 += (x-mean)**2
    
    var = sum2/len(X)
    print 'var: ',var
    
def Q2():
    print "Reading data..."
    data = list(parseData("/C://Users/Khor_000/Desktop/Recommander Sys&Web mining/beer_50000.json"))
    print "done"
    print 'Question 2:'
    bs_count = {}
    beer_styles = [(d['beer/style'],d['review/taste']) for d in data]
    
    
    for bs in beer_styles:
        
        if bs[0] in bs_count.keys():
            bs_count[bs[0]].append(bs[1])
        
        else:
            bs_count[bs[0]] = list([bs[1]])
        
    for style in bs_count.keys():
        print style,":", len(bs_count[style])
        sum1 = 0
        
        for rate in bs_count[style]:
            sum1 += rate
        
        print "avg review for ",style,":", float(sum1)/len(bs_count[style])
        
def Q3():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    print 'Question 3:'
    X = [feature(d) for d in data]
    y = [d['review/taste'] for d in data]
    theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
    
    print 'theta: ', theta
    print 'res: ', residuals
    print 'rank: ', rank
    print 's: ', s
    
    
    
def feature1(datum):
  feat = [1]
  if datum['beer/style'] == "American IPA":
    feat.append(1)
  else:
    feat.append(0)
  return feat

def Q4():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    print 'Question 4:'
    # train
    X = list()
    Y = list()
    Xt = list()
    Yt = list()
    count = 0
    for d in data:
        if count < 25000:
            Xt.append(feature1(d))
            Yt.append(d['review/taste'])
        else:
            X.append(feature1(d))
            Y.append(d['review/taste'])
        count+=1
    theta,res,rank,s = numpy.linalg.lstsq(Xt, Yt)
    theta1,res1,rank1,s1 = numpy.linalg.lstsq(X, Y)
    print 'MSE(train): ', res/25000
    print 'MSE(test): ', res1/25000
    print 'theta_train: ',theta
    print 'theta_test: ',theta1
   
def feature(datum,beer_name):
    feat = [1]
    if datum['beer/style'] == beer_name:
        feat.append(1)
    else:
       feat.append(0)
    return feat    
         
def Q5():
    print "Reading data..."
    data = list(parseData("/C://Users/Khor_000/Desktop/Recommander Sys&Web mining/beer_50000.json"))
    print "done"
    print 'Question 5:' 
    beers = {}
    Y = list()
    # prepare Y
    for d in data:
        Y.append(d['review/taste'])
        if d['beer/style'] not in beers.keys():
            beers[d['beer/style']] = list()
    print 'Y is ready'
    # prepare X
    for d in data:
        beers[d['beer/style']].append([1,1])
        for b in beers.keys():
            if b != d['beer/style']:
                beers[b].append([1,0])
    print 'X is ready'    
    for b in beers.keys():
        xtrain = beers[b][:25000]
        xtest = beers[b][25000:]
        theta,res,rank,s = numpy.linalg.lstsq(xtrain, Y[:25000])
        theta1,res1,rank1,s1 = numpy.linalg.lstsq(xtest, Y[25000:])
        print b,'\'s theta of train:', theta
        print b,'\'s theta of test:', theta1
        print b,'\'s MSE(train): ', res/25000
        print b,'\'s MSE(test): ', res1/25000
        print '--------------------------------'
    
def Q6():
    print "Reading data..."
    data = list(parseData("/C://Users/Khor_000/Desktop/Recommander Sys&Web mining/beer_50000.json"))
    print "done"
    print 'Question 6:'
    #train_data = data[:25000]
    #test_data = data[25000:]
    ran = 20000
    c = 1000
    X = list()
    Y = list()
    for d in data:
        X.append([d['beer/ABV'],d['review/taste']])
        if d['beer/style'] == 'American IPA':
            Y.append(1)
        else:
            Y.append(0)
    print 'XY ready'
    X_train = X[:ran]
    y_train = Y[:ran]
    X_test = X[ran:2*ran]
    y_test = Y[ran:2*ran]
    # Create a support vector classifier object, with regularization parameter C = 1000
    clf = svm.SVC(C=c)
    print 'c = ',c
    clf.fit(X_train, y_train) 
    train_predictions = clf.predict(X_train)
    correct = 0 
    for i in range(0,ran):
        if train_predictions[i] == y_train[i]:
            correct+=1
    print 'train accuracy: ', float(correct)/len(y_train)
    test_predictions = clf.predict(X_test)
    correct = 0
    for i in range(0,ran):
        if test_predictions[i] == y_test[i]:
            correct+=1
    print 'test accuracy: ', float(correct)/len(y_test)

def Q7 ():
    print "Reading data..."
    data = list(parseData("beer_50000.json"))
    print "done"
    print 'Question 7:'
    ran = 20000
    c = 1000
    X = list()
    Y = list()
    for d in data:
        X.append([d['review/taste'],len(d['beer/name'])])
        if d['beer/style'] == 'American IPA':
            Y.append(1)
        else:
            Y.append(0)
    print 'XY ready'
    X_train = X[:ran]
    y_train = Y[:ran]
    X_test = X[ran:2*ran]
    y_test = Y[ran:2*ran]
    # Create a support vector classifier object, with regularization parameter C = 1000
    clf = svm.SVC(C=c)
    print 'c = ',c
    clf.fit(X_train, y_train) 
    train_predictions = clf.predict(X_train)
    correct = 0 
    for i in range(0,ran):
        if train_predictions[i] == y_train[i]:
            correct+=1
    print 'train accuracy: ', float(correct)/len(y_train)
    test_predictions = clf.predict(X_test)
    correct = 0
    for i in range(0,ran):
        if test_predictions[i] == y_test[i]:
            correct+=1
    print 'test accuracy: ', float(correct)/len(y_test)
           
           
        
    
                

        
        