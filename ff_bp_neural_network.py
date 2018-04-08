
# coding: utf-8

# In[1]:

#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'auto')


# In[2]:


from matplotlib import pyplot as plt
import numpy as np


# In[3]:


data = [[3,1.5,1],
        [2,1,0],
        [4,1.5,1],
        [3,1,0],
        [3.5,0.5,1],
        [2,0.5,0],
        [5.5,1,1],
        [1,1,0]]

mystery_flower = [4.5, 1]


# In[4]:


# network
#    o       flower type
#   | \      w1, w2, b
#  o   o     length, width
w1 = np.random.rand()
w2 = np.random.rand()
b  = np.random.rand()


# In[5]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


# In[6]:


T = np.linspace(-6,6,100)
plt.plot(T,sigmoid(T), c='r')
plt.plot(T,sigmoid_derivative(T), c='b')


# In[18]:


# scatter data
plt.axis([0,6,0,6])
plt.grid()
for i in range(len(data)):
    point = data[i]
    color = 'r'
    if point[2] == 0:
        color = 'b'
    plt.scatter(point[0], point[1], c=color)
    
plt.scatter(mystery_flower[0], mystery_flower[1], c='y')


# In[8]:


# training loop

learning_rate = 0.2
costs = []

for i in range(100000):
    random_index = np.random.randint(len(data))
    point = data[random_index]
    
    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)
    
    target = point[2]
    # derivative of your cost function (see how far away we are)
    cost = np.square(prediction - target)

    costs.append(cost)
    
    # derivative part 
    # here we're taking the slope of the cost function (at specific values of b)
    # Cost function:
    dcost_pred = 2 * (prediction - target) #derivative_cost_prediction 
    dpred_dz = sigmoid_derivative(z) # derivative of derivative of prediction (?)
    
    dz_dw1 = point[0] #derivative for w1
    dz_dw2 = point[1] # derivative for w2
    dz_db  = 1 # derivative for b
    
    #dcost_dz = dcost_pred * dpred_dz
    
    # chain rule:
    # get all derivatives of the cost with respective to each of our parameters
    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db  = dcost_pred * dpred_dz * dz_db
    
    # now calculate the learning rate with the slope. 
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
    
plt.plot(costs)


# In[9]:



z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
print(z)
pred = sigmoid(z)
pred


# In[27]:


import os

def which_flower(length, width):
    z = length * w1 + width * w2 + b
    pred = sigmoid(z)
    if pred < .5:
        print([length,width],'->', pred)
        os.system('espeak "the mystery flower is blue"')        
        plt.scatter(length,width, c='b')
    else:
        print([length,width], '->', pred)
        os.system('espeak "the mystery flower is red"')
        plt.scatter(length,width, c='r')
        


# In[28]:


which_flower(mystery_flower[0], mystery_flower[1])


# In[35]:


which_flower(np.inf,np.inf)


# In[12]:


# plt.axis([-50,50,-50,50])
# plt.grid()

z = 0
pred = 0

def predict_random(rows):     
    # generate random flower data:
    flowers = []
    for i in range(rows): 
        flowers.append([np.random.randint(-10, 10),
         np.random.randint(-10, 10),
         0])
    
    for i in flowers:        
        l = i[0]
        w = i[1]    
        z = l * w1 + w * w2 + b
        pred = sigmoid(z)
        i[2] = pred    
        if pred < 0.5:
            print([l,w],'->', pred, '->', 'blue')
            plt.scatter(l,w, c='b')
        else:
            print([l,w], '->', pred, '->','red')
            plt.scatter(l,w, c='r')        
            
predict_random(20)

# In[13]:
print(6.771092766813517e-42 < 0.5)

