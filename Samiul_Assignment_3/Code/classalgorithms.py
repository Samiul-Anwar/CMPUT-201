from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import scipy.spatial as sp

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0
        self.yprob = []

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
       
        self.numfeatures  = Xtrain.shape[1]
        numsamples = Xtrain.shape[0]
        #print (self.numfeatures)
        count = 0
        for i in ytrain:
            if (i>count):
                count+=1
        self.numclasses = count + 1
           
        if(self.params['usecolumnones']==False):
           b = np.ones((numsamples, self.numfeatures-1))
           b = Xtrain[:,:-1]
           Xtrain = b
           self.numfeatures -= 1
        # print(Xtrain.shape[1])

        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        countclass = np.zeros(self.numclasses)
        for i in range (0, numsamples):
            k = int(ytrain[i])
            countclass[k] += 1
            for j in range (0, self.numfeatures):
                self.means[k][j]+=Xtrain[i][j]
    
        for i in range (0, self.numclasses):
        #np.true_divide(self.means[i], countclass[i])
            for j in range (0, self.numfeatures):
                self.means[i][j] = self.means[i][j]/(countclass[i]+1e-8)
        
        self.yprob = np.true_divide(countclass, numsamples)
        
        for i in range (0, numsamples):
            k = int(ytrain[i])
            for j in range (0, self.numfeatures):
                self.stds[k][j]+= (Xtrain[i][j] - self.means[k][j])**2
                # print (self.stds)
        
        for i in range (0, self.numclasses):
        #np.true_divide(self.stds[i], countclass[i])
            for j in range (0, self.numfeatures):
                self.stds[i][j] = self.stds[i][j]/(countclass[i]+1e-8)
        
        #  print (self.means)
        #  print (self.stds)
        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        #print (self.yprob)
        if(self.params['usecolumnones']==False):
            b = np.ones((Xtest.shape[0], self.numfeatures))
            b = Xtest[:,:-1]
            Xtest = b
        ### YOUR CODE HERE
        for i in range (0, Xtest.shape[0]):
            bestval =0
            ycls = np.ones(self.numclasses)
            for x in range (0, self.numclasses):
                ycls[x] = ycls[x]*self.yprob[x]
                    #print (ycls)
            for j in range (0, self.numfeatures):
                for k in range (0, self.numclasses):
                    left = (2 * np.pi* self.stds[k][j] )
                    left = np.power(left, -0.5)
                    right = -(Xtest[i][j]-self.means[k][j])**2
                    right = right/(2*self.stds[k][j])
                    right = np.exp(right)
                    # if(j==8 & i<2):
                        #   print (left)
                        #print (right)
                        # print (ycls[k] * left * right)
                        #  ycls[k] *= ((2 * np.pi* self.stds[k][j] )**-.5 ) * np.exp((-(Xtest[i][j]-self.means[k][j])**2)/(2*self.stds[k][j]))
                    ycls[k] = ycls[k] * left * right


            #  print (ycls)
                        #print (ycls)
            # print (self.yprob)
            #ycls = np.multiply(ycls, self.yprob)
            # print (ycls)
                #for i in range (0, self.numclasses):
                #ycls[i] = ycls[i]*self.yprob[i]
                #   print (ycls)
                #   print (np.argmax(ycls))
            ytest[i] = np.argmax(ycls)
        
        
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'lamb' : 0.001, 'stepsize': 0.001}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE
        sig = utils.sigmoid(theta)
        
        for i in range(0, X.shape[0]):
            cost += (y[i]-1)*theta[i] + np.log(sig[i])
        ### END YOUR CODE
        cost = cost #+ 0.01 * self.regularizer[0](self.weights)
        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        sig = utils.sigmoid(theta)
        # sig = np.subtract(sig, y)
        sig = sig - y
        grad = np.dot(X.T, sig) + 2 * self.params['lamb'] * self.regularizer[1](self.weights)
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        pass
        self.weights = np.zeros(Xtrain.shape[1],)

        ### YOUR CODE HERE
        
        lmbd = self.params['lamb']
        
        numsamples = Xtrain.shape[0]
        # Xless = Xtrain[:,self.params['features']]
        Xless = Xtrain
        self.weights = np.random.rand(Xless.shape[1])
        err = 10000;
        #cw =0;
        tolerance = 10*np.exp(-4)
        i=0;
        
        
        w1 = self.weights
        #     cw_v =(np.dot(Xless, self.weights)-ytrain)
        #cw = (np.linalg.norm(cw_v)**2)/(2*numsamples)
        cw_v = np.dot(Xless, self.weights.T)
        cw = self.logit_cost(cw_v, Xless, ytrain) + lmbd * self.regularizer[0](self.weights)
        #  print(cw)
        errors = []
        runtm = []
        epch = []
        
        err = 1
        iteration= 1000
        #tm= time.time()
        while (abs(cw-err)>tolerance) and (i <iteration):
            err = cw
            g =  self.logit_cost_grad(cw_v, Xless, ytrain)
            obj = cw
            j=0
            ita = -1* self.params['stepsize']
            w = self.weights
            #  w1 = np.add(w,np.dot(ita,g))
            while(j<iteration):
                w1 = np.add(w,np.dot(ita,g))
                #  cw_v =(np.dot(Xless, w1)-ytrain)
                # cw = (np.linalg.norm(cw_v)**2)/(2*numsamples)
                cw_v = np.dot(Xless, w1.T)
                cw = self.logit_cost(cw_v, Xless, ytrain)+lmbd * self.regularizer[0](w1)
                ##    print (cw)
                
                if(cw<np.absolute(obj-tolerance)):  ############################################
                    break
                ita = 0.7*ita
                j=j+1
            
            if(j==iteration):
                self.weights=w
                ita =0
            else:
                self.weights = w1
            
            # cw_v =(np.dot(Xless, self.weights)-ytrain)
            #cw = (np.linalg.norm(cw_v)**2)/(2*numsamples)
            cw_v = np.dot(Xless, self.weights.T)
            cw = self.logit_cost(cw_v, Xless, ytrain)
            #tm1 = time.time()-tm
            #runtm.append(tm1)
            #err = cw
            errors.append(err)
            i=i+1
            epch.append(i)

#  print(self.weights)
        
        
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        sig = np.dot(Xtest, self.weights)
        sig = utils.sigmoid(sig)
        #print (sig)
        sig = np.round(sig)
        #print (sig)
        for i in range (0, ytest.shape[0]):
            ytest[i] = int(sig[i])
        ### END YOUR CODE
        #print (ytest)
        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 100}
        self.reset(parameters)
    
    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None
        self.ahidden = None
        self.aout = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        # a_hidden = self.transfer(np.dot(self.w_input, inputs))
        a_hidden = self.transfer(np.dot(inputs, self.w_input))
        
        #a_output = self.transfer(np.dot(self.w_output, a_hidden))
        dots =  (np.dot(a_hidden, self.w_output))
        a_output = self.transfer(np.asarray(dots))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        nabla_output = np.zeros(self.params['nh'])
        del1 = self.aout - y
        nabla_output = del1 * self.ahidden
        #print (nabla_output)
        a1 = del1 * self.w_output
        a2 = np.multiply(self.ahidden, (1-self.ahidden))
        del2 = np.zeros(self.params['nh'])
        for i in range(0, self.params['nh']):
            del2[i] = self.w_output[i]* del1 * self.ahidden[i]*(1-self.ahidden[i])
        #  del2 = np.multiply(a1,a2)
        # nabla_input = np.multiply(del2, x)
        nabla_input = np.zeros([x.shape[0],self.params['nh']])
        for i in range (0, x.shape[0]):
            for j in range(0, self.params['nh']):
                nabla_input[i][j] = x[i]*del2[j]
            #nabla_input = np.dot(x, np.transpose(del2))
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):



        numsamples = Xtrain.shape[0]
        Xless = Xtrain
        self.w_input = np.random.rand(Xless.shape[1], self.params['nh'])
        self.w_output= np.random.rand(self.params['nh'])
        self.ahidden = np.zeros(self.params['nh'])
        
        
        
        #Xless=Xtrain
        ita = -0.1
        

        for i in range (0, self.params['epochs']):
            randomize = np.arange(len(ytrain))
            np.random.shuffle(randomize)
            Xless = Xless[randomize]
            ytrain = ytrain[randomize]
        
            for j in range (0, numsamples):
                self.ahidden, self.aout = self.feedforward(Xless[j])
                g_input, g_output = self.backprop(Xless[j], ytrain[j])
                self.w_input = np.add(self.w_input, ita * g_input)
                self.w_output = np.add(self.w_output , ita * g_output)
                    # print (self.w_output)
#print (self.w_input)


    def predict(self, Xtest):

        ytest = np.zeros(Xtest.shape[0])
        for j in range (0, Xtest.shape[0]):
            a, b = self.feedforward(Xtest[j])
            #print (self.aout)
            ytest[j] = int (np.round (b))
        assert len(ytest) == Xtest.shape[0]
        return ytest

















class NeuralNet2(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
        used as the cost function.
        
        Parameters:
        nh -- number of hidden units
        transfer -- transfer function, in this case, sigmoid
        stepsize -- stepsize for gradient descent
        epochs -- learning epochs
        
        Note:
        1) feedforword will be useful! Make sure it can run properly.
        2) Implement the back-propagation algorithm with one layer in ``backprop`` without
        any other technique or trick or regularization. However, you can implement
        whatever you want outside ``backprob``.
        3) Set the best params you find as the default params. The performance with
        the default params will affect the points you get.
        """
    def __init__(self, parameters={}):
        self.params = {'nh1': 16,
            'nh2' : 16,
            'transfer': 'sigmoid',
                'stepsize': 0.01,
                    'epochs': 100}
        self.reset(parameters)
    
    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_middle = None
        self.w_output = None
        self.ahidden1 = None
        self.ahidden2 = None
        self.aout = None
    
    def feedforward(self, inputs):
        """
            Returns the output of the current neural network for the given input
            """
        # hidden activations
        # a_hidden = self.transfer(np.dot(self.w_input, inputs))
        a_hidden1 = self.transfer(np.dot(inputs, self.w_input))
        
        dots1 =  (np.dot(a_hidden1, self.w_middle))
        a_hidden2 = self.transfer(np.asarray(dots1))
        
        #a_output = self.transfer(np.dot(self.w_output, a_hidden))
        dots2 =  (np.dot(a_hidden2, self.w_output))
        a_output = self.transfer(np.asarray(dots2))
        
        return (a_hidden1, a_hidden2, a_output)
    
    def backprop(self, x, y):
        """
            Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
            for the cost function with respect to self.w_input and self.w_output.
            """
        
        ### YOUR CODE HERE
        nabla_output = np.zeros(self.params['nh2'])
        del1 = self.aout - y
        nabla_output = del1 * self.ahidden2
        #print (nabla_output)
        del2 = np.zeros(self.params['nh2'])
        for i in range(0, self.params['nh2']):
            del2[i] = self.w_output[i] * del1 * self.ahidden2[i] * (1-self.ahidden2[i])
        #  del2 = np.multiply(a1,a2)
        # nabla_input = np.multiply(del2, x)
        nabla_middle = np.zeros([self.params['nh1'],self.params['nh2']])
        for i in range (0, self.params['nh1']):
            for j in range(0, self.params['nh2']):
                #nabla_middle[i][j] = self.ahidden2[j] * del2[j]
                nabla_middle[i][j] = self.ahidden1[i] * del2[j]
        
        #nabla
        del3 = np.zeros(self.params['nh1'])
        #  del3 = np.dot(self.w_middle, del2)
        for i in range(0, self.params['nh1']):
            del3[i] = np.dot(self.w_middle[i], del2) * self.ahidden1[i] * (1-self.ahidden1[i])
        #  del2 = np.multiply(a1,a2)
        # nabla_input = np.multiply(del2, x)
        nabla_input = np.zeros([x.shape[0],self.params['nh1']])
        for i in range (0, x.shape[0]):
            for j in range(0, self.params['nh1']):
                #nabla_input[i][j] = self.ahidden1[j]*del3[j]
                nabla_input[i][j] = x[i]*del3[j]
        
        
            #nabla_input = np.dot(x, np.transpose(del2))
        ### END YOUR CODE
        
        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_middle, nabla_output)

# TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
    
    
    
        numsamples = Xtrain.shape[0]
        Xless = Xtrain
        self.w_input = np.random.rand(Xless.shape[1], self.params['nh1'])
        self.w_middle = np.random.rand(self.params['nh1'], self.params['nh2'])
        self.w_output= np.random.rand(self.params['nh2'])
        self.ahidden1 = np.zeros(self.params['nh1'])
        self.ahidden2 = np.zeros(self.params['nh2'])
        
        
        
        #Xless=Xtrain
        ita = -1* self.params['stepsize']


        for i in range(1, self.params['epochs']+1):
            meansq = 0
            randomize = np.arange(len(ytrain))
            np.random.shuffle(randomize)
            Xless = Xless[randomize]
            ytrain = ytrain[randomize]
            ita = ita/i
            for j in range(0, numsamples):
                self.ahidden1,self.ahidden2, self.aout = self.feedforward(Xless[j])
                g_input, g_middle, g_output = self.backprop(Xless[j], ytrain[j])
                g1 = np.zeros([Xless.shape[1], self.params['nh1']])
                g2 = np.zeros([self.params['nh1'], self.params['nh2']])
                g3 = np.zeros(self.params['nh2'])
                meansq1 = np.zeros([Xless.shape[1], self.params['nh1']])
                meansq2 = np.zeros([self.params['nh1'], self.params['nh2']])
                meansq3 = np.zeros(self.params['nh2'])
                
                
                g1 = g_input**2
                g2 = g_middle**2
                g3 = g_output**2
                meansq1 = 0.9*meansq1 + 0.1 * g1
                meansq2 = 0.9*meansq2 + 0.1 * g2
                meansq3 = 0.9*meansq3 + 0.1 * g3
                
                self.w_input = np.add(self.w_input,  ((ita/((meansq1+0.0001)**0.5))*g_input))
                self.w_middle = np.add(self.w_middle, (ita/((meansq2+0.0001)**0.5)*g_middle))
                self.w_output = np.add(self.w_output, (ita/((meansq3+0.0001)**0.5)*g_output))


        
    # print (self.w_output)
#print (self.w_input)


    def predict(self, Xtest):
    
        ytest = np.zeros(Xtest.shape[0])
        for j in range (0, Xtest.shape[0]):
            a, x, b = self.feedforward(Xtest[j])
            #print (b)
            ytest[j] = int (np.round (b))
        assert len(ytest) == Xtest.shape[0]
        return ytest






class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None', 'k' : 10, 'lamb': 0.001, 'stepsize': 0.001}
        self.reset(parameters)
    def reset(self, parameters):
        self.kcentre = None
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))

    def hamming (self, x1, y1):
        return sum(el1 != el2 for el1, el2 in zip(x1, y1))

    
    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None
        
        ### YOUR CODE HERE
        Xless = Xtrain
        randomize = np.arange(len(ytrain))
        np.random.shuffle(randomize)
        Xless = Xless[randomize]
        self.kcentre = Xless[:self.params['k']]
        #ytrain = ytrain[randomize]
        
        # print (Xtrain.shape)
        if (self.params['kernel'] == 'hamming'):
            print('')
            Ktrain = np.zeros([Xtrain.shape[0], self.params['k']])
            for i in range (0, Xtrain.shape[0]):
                for j in range (0, self.params['k']):
                    Ktrain[i][j] = (self.hamming(Xtrain[i], self.kcentre[j]))
        
        else:
            
            Ktrain = np.dot(Xtrain, self.kcentre.T)
        #print (self.kcentre.shape)
        ### END YOUR CODE
       
        self.weights = np.zeros(Ktrain.shape[1])

        ### YOUR CODE HERE
        super(KernelLogitReg , self).learn(Ktrain, ytrain)
        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def predict(self, Xtest):
        """
            Use the parameters computed in self.learn to give predictions on new
            observations.
            """
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        if (self.params['kernel'] == 'hamming'):
            print('')
            Ktest = np.zeros([Xtest.shape[0], self.params['k']])
            for i in range (0, Xtest.shape[0]):
                for j in range (0, self.params['k']):
                    Ktest[i][j] = self.hamming(Xtest[i], self.kcentre[j])
        
        
        else:
            
            Ktest = np.dot(Xtest, self.kcentre.T)
        ### YOUR CODE HERE
        sig = np.dot(Ktest, self.weights)
        sig = utils.sigmoid(sig)
        #print (sig)
        sig = np.round(sig)
        #print (sig)
        for i in range (0, ytest.shape[0]):
            ytest[i] = int(sig[i])
        ### END YOUR CODE
        #print (ytest)
        assert len(ytest) == Xtest.shape[0]
        return ytest


# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
