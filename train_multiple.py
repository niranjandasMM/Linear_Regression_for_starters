import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D


def mse(y_pred, label):
    # print(f"y_pred is : {y_pred} and label is : {label}")
    return np.mean((y_pred-label)**2)

class MultipleLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
    
    def optimize(self, x,  y_pred, y_train, learning_rate):
        error = [ (y_train[i] - y_pred[i] ) for i in range(len(x)) ]

        for i in range(len(x)):
            self.b1 += learning_rate * x[i][0] * error[i]
            self.b2 += learning_rate * x[i][1] * error[i]
            self.b3 += learning_rate * x[i][2] * error[i]
            self.b0 += learning_rate * error[i]

        # print(f"self.b0 and self.b1 are : {self.b0, self.b1, self.b2, self.b3}")

    def predict(self,x):
        y_pred = [self.b0 + (self.b1 * xi[0]) + (self.b2 * xi[1]) + (self.b3 * xi[2]) for xi in x]
        return y_pred
        
    def train(self, x_train, y_train, epochs):
        loss_list = []

        for epoch in range(epochs):

            y_pred = self.predict(x_train)

            loss = mse(y_pred, y_train)
            loss_list.append( loss )

            self.optimize(x_train, y_pred, y_train, learning_rate=0.001)
            
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

        # # Scatter plot - Target vs Each Feature
        plt.scatter(x_train[:, 0], y_train, color="blue", label="Actual feature 1")
        plt.scatter(x_train[:, 0], y_pred, color="red", label="Predicted feature 1");
        plt.xlabel("Feature 1")
        plt.ylabel("Target")
        plt.legend()
        plt.show()

        plt.scatter(x_train[:, 1], y_train, color="blue", label="Actual feature 2")
        plt.scatter(x_train[:, 1], y_pred, color="red", label="Predicted feature 2")
        plt.xlabel("Feature 2")
        plt.ylabel("Target")
        plt.legend()
        plt.show()

        plt.scatter(x_train[:, 2], y_train, color="blue", label="Actual fetaure 3")
        plt.scatter(x_train[:, 2], y_pred, color="red", label="Predicted feature 3")
        plt.xlabel("Feature 3")
        plt.ylabel("Target")
        plt.legend()
        plt.show()

        
        ## It is commented right now because its hard to visaualize a 3D or 4D graph
        # # Scatter plot of actual and predicted values 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=y_train, cmap='viridis', label='Actual')
        # ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=y_pred, cmap='inferno', marker='x', s=50, label='Predicted')

        # # Create a meshgrid of feature values
        # feature1_range = np.linspace(min(x_train[:, 0]), max(x_train[:, 0]), num=10)
        # feature2_range = np.linspace(min(x_train[:, 1]), max(x_train[:, 1]), num=10)
        # feature3_range = np.linspace(min(x_train[:, 2]), max(x_train[:, 2]), num=10)
        # feature1_mesh, feature2_mesh, feature3_mesh = np.meshgrid(feature1_range, feature2_range, feature3_range)

        # # Calculate predicted values for the meshgrid
        # x_mesh = np.column_stack((feature1_mesh.flatten(), feature2_mesh.flatten(), feature3_mesh.flatten()))
        # y_mesh = self.predict(x_mesh)  # Replace "self.predict" with your prediction function

        # # Convert lists to NumPy arrays and reshape
        # feature1_mesh = np.array(feature1_mesh).reshape(-1)
        # feature2_mesh = np.array(feature2_mesh).reshape(-1)
        # feature3_mesh = np.array(feature3_mesh).reshape(-1)
        # y_mesh = np.array(y_mesh).reshape(-1)

        # # Plot the best fit line or hyperplane
        # ax.plot_trisurf(feature1_mesh, feature2_mesh, feature3_mesh, cmap='coolwarm', edgecolor='none', alpha=0.5)
        # ax.scatter(x_mesh[:, 0], x_mesh[:, 1], x_mesh[:, 2], c=y_mesh, cmap='coolwarm', marker='.', label='Best Fit')

        # ax.set_xlabel('Feature 1')
        # ax.set_ylabel('Feature 2')
        # ax.set_zlabel('Feature 3')
        # ax.set_title('Scatter Plot of Features with Best Fit Line/Plane')
        # ax.legend()
        # plt.show()
