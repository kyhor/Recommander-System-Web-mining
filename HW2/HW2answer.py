# -*- coding: utf-8 -*-
import numpy
from urllib import urlopen
import scipy.optimize
import random
from math import exp
from math import log

def parseData(fname):
  for l in urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("beer_50000.json"))
print("done")

def feature(datum):
  feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
  return feat

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
  return numpy.array([-x for x in dl])

##################################################
# Train                                          #
##################################################

def train(lam,X,X_train,y_train):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta,X,y_train):
  scores = [inner(theta,x) for x in X]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y_train)]
  acc = sum(correct) * 1.0 / len(correct)
  return acc

def Q1():
    print "Question 1:"
    X_train = [feature(d) for d in data[:len(data)/3] ]
    X_valid = [feature(d) for d in data[len(data)/3:(len(data)*2)/3] ]
    X_test = [feature(d) for d in data[(len(data)*2)/3:] ]
    y_train = [d['beer/ABV'] >= 6.5 for d in data[:len(data)/3]]
    y_valid = [d['beer/ABV'] >= 6.5 for d in data[len(data)/3:(len(data)*2)/3]]
    y_test = [d['beer/ABV'] >= 6.5 for d in data[(len(data)*2)/3:]]
    lam = 1.0
    theta = train(lam,X_train,X_train,y_train)
    acc = performance(theta,X_valid,y_valid)
    print "valid acc: ", acc
    acc = performance(theta,X_test,y_test)
    print "test acc: ", acc
 
print "Question 2:"   
def Q2feature(datum):
    wordcount = { 'lactic':0, 'tart':0, 'sour':0, 'citric':0, 'sweet':0, 'acid':0, 'hop':0, 'fruit':0, 'salt':0, 'spicy':0}
    feat  = [1]
    review = datum['review/text']
    review = review.lower()
    review = review.split(' ')
    for word in review:
        if not word[len(word)-1].islower():
            word = word[:len(word)-1]
        if word in wordcount.keys():
            wordcount[word] += 1
    for w in wordcount.keys():
        feat.append(wordcount[w])
    print feat
    return feat

def Q3():
    print 'Question 3:'
    FN = float(0)
    TN = float(0)
    TP = float(0)
    FP = float(0)
    X_train = [feature(d) for d in data[:len(data)/3] ]
    y_train = [d['beer/ABV'] >= 6.5 for d in data[:len(data)/3]]
    X_test = [feature(d) for d in data[(len(data)*2)/3:] ]
    y_test = [d['beer/ABV'] >= 6.5 for d in data[(len(data)*2)/3:]]
    lam = 1.0
    theta = train(lam,X_train,X_train,y_train)
    scores = [inner(theta,x) for x in X_test]
    predictions = [s > 0 for s in scores]
    for (p,r) in zip(predictions,y_test):
        if p and r:
            TP +=1 
        elif p == 1 and r == 0:
            FP +=1
        elif p == 0 and r==1:
            FN += 1
        elif p == 0 and r == 0:
            TN+=1
    BER =  0.5*(FP/(TN+FP) + FN/(FN+TP))
    print 'TN:',TN
    print 'TP:',TP
    print 'FN:',FN
    print 'FP:',FP
    print 'balanced error rate: ', BER
    
    
