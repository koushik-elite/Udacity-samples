# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('../student_data.csv')

# Printing out the first 10 rows of our data
# data[:10]

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
# Plotting the points
# plot_points(data)
# plt.show()

# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
# plot_points(data_rank1)
# plt.title("Rank 1")
# plt.show()
# plot_points(data_rank2)
# plt.title("Rank 2")
# plt.show()
# plot_points(data_rank3)
# plt.title("Rank 3")
# plt.show()
# plot_points(data_rank4)
# plt.title("Rank 4")
# plt.show()

one_hot_data = pd.get_dummies(data, columns=['rank'])
print(one_hot_data.drop('rank', axis=1));