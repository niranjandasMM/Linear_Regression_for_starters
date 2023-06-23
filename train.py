import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def mse(y_pred, label):
    # print(f"y_pred is : {y_pred} and label is : {label}")
    return np.mean((y_pred-label)**2)

class LinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
    
    def optimize(self, x_train, y_train, learning_rate):
        x = x_train
        predicted_price = [(self.b1 * xi) + self.b0 for xi in x]
        error = [ (y_train[i] - predicted_price[i] ) for i in range(len(x)) ]

        for i in range(len(x)):
            self.b1 += learning_rate * x[i] * error[i]
            self.b0 += learning_rate * error[i]
        # print(f"self.b0 and self.b1 are : {self.b0, self.b1}")

    def predict(self,x):
        y_pred = [ (self.b1 * xi) + self.b0 for xi in x ] # The formula y^ = b0 + (b1* x)
        # plt.scatter(x, y_pred, color="pink", label="best fit regression line") #reg_line
        return y_pred
        
    def train(self, x_train, y_train, epochs):
        loss_list = []

        for epoch in range(epochs):
            y_pred = self.predict(x_train)

            loss = mse(y_pred, y_train)
            loss_list.append( loss )

            self.optimize(x_train, y_train, learning_rate=0.01)
            
            if epoch % 10 == 0:
                sys.stdout.write(
                    "\n" +
                    "I:" + str(epoch) +
                    " Train-Err:" + str(loss / float(len(x_train)))[0:5] +
                    "\n"
                )

        for x, y in zip(y_pred, y_train):
            df = pd.DataFrame({"y_pred": [x], "y_train": [y]})
            # print(df)


        r2 = r2_score(y_train, y_pred)
        print("R2 Score:", r2)
        print(y_pred) 

        plt.plot(loss_list) ; plt.show()
        plt.scatter(x_train, y_train, color="blue", label="x axis and y axis") #our data points
        plt.scatter(x_train, y_pred, color="pink", label="best fit regression line") #reg_line ;
        plt.legend()
        plt.show()

    