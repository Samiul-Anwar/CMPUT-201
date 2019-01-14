from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))

## k-fold cross-validation
# K - number of folds
# X - data to partition
# Y - targets to partition
# classalgs - a dictionary mapping algorithm names to algorithm instances
#
# example:
# classalgs = {
#   'nn_0.01': algs.NeuralNet({ 'regwgt': 0.01 }),
#   'nn_0.1':  algs.NeuralNet({ 'regwgt': 0.1  }),
# }

def STDeviation (er, m):
    d = 0.0
    for i in er:
        k = float(i-m)
        d = d + (i-m)**2
    
    d = d/(len(er)-1)
    d = d**0.5
    return d

def cross_validate(K, X, Y, classalgs):
    errors={}
    std = {}
    part = int(X.shape[0]/K)
    for learnername, learner in classalgs:
        errors[learnername]=0
        std[learnername] = np.zeros(K)
    for k in range(K):
        for learnername, learner in classalgs:
            #learner.reset(params)
            xtest = X[k*K:k*K+part-1]
            ytest = Y[k*K:k*K+part-1]
            xtrain = np.delete(X, slice(k*K,k*K+part-1), axis=0)
            ytrain = np.delete(Y, slice(k*K,k*K+part-1), axis=0)
            print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
            learner.learn(xtrain, ytrain)
                # Test model
            predictions = learner.predict(xtest)
            error = geterror(ytest, predictions)
            std[learnername][k] = error
            print ('Error for ' + learnername + ': ' + str(error))
            errors[learnername] += error/K

#errors = errors/K
# errors = {i: v / K for i, v in errors}
# errors = dict((key, [x / K for x in values]) for key, values in errors.items())
#   for key in errors:
#        errors[key] = errors
    for learnername, learner in classalgs:
        print ('Average Error for ' + learnername + ': ' + str(errors[learnername]))
        print ('Standard Deviation of Errrors ' + learnername + ': ' + str(STDeviation(std[learnername], errors[learnername])))

    learner = min(errors, key=errors.get)
    
    best_algorithm = learner
    return best_algorithm



def strat_cross_validate(K, X, Y, classalgs):
    errors={}
    std = {}
    clss = np.unique(Y)
    clsA = []
    clsB = []
   
    for i in range (0, X.shape[0]):
        if (Y[i]==clss[0]):
            clsA.append(i)
        else:
            clsB.append(i)

#part = int(X.shape[0]/K)
    for learnername, learner in classalgs:
        errors[learnername]=0
        std[learnername] = np.zeros(K)
    for k in range(K):
        for learnername, learner in classalgs:
            #learner.reset(params)
            xtest = np.zeros([int(X.shape[0]/K), X.shape[1]])
            ytest = np.zeros(int(Y.shape[0]/K))
            xtrain = np.zeros([X.shape[0]-int(X.shape[0]/K), X.shape[1]])
            ytrain = np.zeros(Y.shape[0] - int(Y.shape[0]/K))
            i=0
            j=0
            l=0
            while(l < len(clsA)):
                if(l >= k*K and l< (k*K+ int(len(clsA)/K))):
                    x = clsA[l]
                    xtest[i] = X[x]
                    ytest[i] = Y[x]
                    i += 1
                
                else:
                    xtrain[j] = X[clsA[l]]
                    ytrain[j] = Y[clsA[l]]
                    j += 1
                
                l += 1
            l=0
            #  j= j
            
            while(l < len(clsB) and j < (X.shape[0]-int(X.shape[0]/K)) and i < (int(Y.shape[0]/K))):
                
                if(l >= k*K and l< (k*K+ int(len(clsB)/K))):
                    xtest[i] = X[clsB[l]]
                    ytest[i] = Y[clsB[l]]
                    i += 1
                
                else:
                    
                    xtrain[j] = X[clsB[l]]
                    ytrain[j] = Y[clsB[l]]
                    j += 1
                
                l += 1

            print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
            # Train model
            learner.learn(xtrain, ytrain)
            # Test model
            predictions = learner.predict(xtest)
            error = geterror(ytest, predictions)
            std[learnername][k] = error
            print ('Error for ' + learnername + ': ' + str(error))
            errors[learnername] += error/K

#errors = errors/K
# errors = {i: v / K for i, v in errors}
# errors = dict((key, [x / K for x in values]) for key, values in errors.items())
#   for key in errors:
#        errors[key] = errors
# print (std)
    for learnername, learner in classalgs:
        print ('Average Error for Stratified Cross Validation' + learnername + ': ' + str(errors[learnername]))
        print ('Standard Deviation of Errrors for Stratified Cross Validation ' + learnername + ': ' + str(STDeviation(std[learnername], errors[learnername])))
    
    learner = min(errors, key=errors.get)
    
    best_algorithm = learner
    return best_algorithm



if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 3

    classalgs = {#'Random': algs.Classifier(),
        'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
             'Naive Bayes Ones': algs.NaiveBayes({'usecolumnones': True}),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Logistic Regression Reg': algs.LogitReg({'regularizer':'l2', 'lamb' : 0.001, 'stepsize' : 0.001}),
                 'Logistic Regression': algs.LogitReg({'lamb' : 0.001, 'stepsize': 0.001}),
                     'kernel Logistic Regression': algs.KernelLogitReg({'k':30}),
                    'Hamming kernel Logistic Regression': algs.KernelLogitReg({'kernel': 'hamming','k':20}),
                     'Neural Network': algs.NeuralNet({'epochs': 100}),
                    'Neural Network2': algs.NeuralNet2({'epochs': 100})
                }
    numalgs = len(classalgs)


    cls = {
         'Logistic RegressionRegularized': algs.LogitReg({'regularizer':'l2', 'lamb' : 0.001, 'stepsize': 0.001}),
         'Logistic RegressionRegularized': algs.LogitReg({'regularizer':'l2', 'lamb' : 0.001, 'stepsize': 0.01}),
         'Logistic RegressionRegularized': algs.LogitReg({'regularizer':'l2', 'lamb' : 0.001, 'stepsize': 0.1}),
         'Logistic Regression': algs.LogitReg({'stepsize': 0.001}),
         'Logistic Regression': algs.LogitReg({'stepsize': 0.01}),
         'Logistic Regression': algs.LogitReg({'stepsize': 0.1}),
         'Neural Network': algs.NeuralNet2({'nh': 8}),
         'Neural Network': algs.NeuralNet2({'nh': 16}),
         'Neural Network': algs.NeuralNet2({'nh': 32})
            }


    parameters = (
        {'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        {'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)
    trainset, testset = dtl.load_susy(trainsize,testsize)
    learnername = cross_validate(3, trainset[0], trainset[1], cls.items() )
    print ('Best parameters for k fold ' + learnername )

    learnername = strat_cross_validate(3, trainset[0], trainset[1], cls.items() )

    print ('Best parameters for Stratified k fold ' + learnername )
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset1, testset1 = dtl.load_susy(trainsize,testsize)
        #trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        trainset2, testset2 = dtl.load_census(trainsize,testsize)
    
    
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters;
                if (learnername == 'Hamming kernel Logistic Regression'):
                    trainset = trainset2
                    testset = testset2
                else:
                    trainset = trainset1
                    testset = testset1
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
