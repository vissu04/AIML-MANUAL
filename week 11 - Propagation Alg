import numpy as np
X = np.array(([2,9],[1,5],[3,6])) #Hours Studied,Hours Slept
y=np.array(([92],[86],[89])) #Test Score

y=y/100 #Max Test Score is 100

#Sigmoid Function
def sigmoid(x):
  return 1/(1+ np.exp(-x))

#Derivatives of Sigmoid function
def derivatives_sigmoid(x):
  return x*(1-x)
#Variable initialization
epoch=10000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of features in data set
hiddenlayers_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons of output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayers_neurons))
bias_hidden=np.random.uniform(size=(1,hiddenlayers_neurons))  #bias matrix to the hidden layer
weight_hidden=np.random.uniform(size=(hiddenlayers_neurons,output_neurons)) #weight matrix to the output layer
bias_output=np.random.uniform(size=(1,output_neurons)) #matrix to output layer

for i in range(epoch):
  hinp1=np.dot(X,wh)
  hinp=hinp1+ bias_hidden
  hlayer_activation = sigmoid(hinp)

  outinp1=np.dot(hlayer_activation,weight_hidden)
  outinp = outinp1+bias_output
  output = sigmoid(outinp)

EO = y-output
outgrad=derivatives_sigmoid(output)
d_output = EO * outgrad
EH = d_output.dot(weight_hidden.T)
hiddengrad=derivatives_sigmoid(hlayer_activation)
d_hiddenlayer = EH * hiddengrad

weight_hidden += hlayer_activation.T.dot(d_output) * lr
bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) * lr
wh += X.T.dot(d_hiddenlayer) * lr
bias_output += np.sum(d_output,axis=0,keepdims=True) *lr

print("Input: \n"+str(X))
print("Actual Output: \n"+str(y))
print("Predicted Output: \n",output)
