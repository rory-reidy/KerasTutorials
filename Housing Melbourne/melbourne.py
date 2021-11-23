import pandas as pd 
#import decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#import mean absolute error
from sklearn.metrics import mean_absolute_error
#can break up the data into two pieces
from sklearn.model_selection import train_test_split

mdata = pd.read_csv("melb_data.csv")

# dropna drops rows with missing values (think of na as "not available")
mdata = mdata.dropna(axis=0)

#selecting prediction target:
y = mdata.Price

#selecting multiple features:
#list of column names:
mfeatures = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude']
X = mdata[mfeatures]
#SPLIT DATA INTO TRAINING AND TESTING
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y, forest):
    if forest:
        model = RandomForestRegressor(random_state=1)
    else:
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
print("DecisionTreeRegressor:")
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y, False)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
print("RandomForestRegressor:")
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y, True)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

"""
PRINTING
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(model.predict(X.head()))
PREDICTING ALL PRICE
pred_price = model.predict(X)
print(mean_absolute_error(y, pred_price))


val_pred = model.predict(val_X)
print(mean_absolute_error(val_y, val_pred))
"""