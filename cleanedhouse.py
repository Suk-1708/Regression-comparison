import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

importdf = pd.read_csv("housing.csv")  #reading file 
df = importdf.dropna()    #removing any empty values
#print(df.isnull().sum().sum())  # test to see if there is any  empty values remaining

df = df.sample(frac=1) #randomize the data every execution
train_size = int(0.8 * len(df)) # splitting data into train:test, 80%:20%
trainset = df[:train_size] #assigning the training data to set
k = 5
folds = np.array_split(trainset,k)
testset =  df[train_size:] #assigning the testing data to set

rooms = trainset["Avg. Area Number of Rooms"].tolist() #converting the x-column into a list in the 80% data -Training
price = trainset["Price"].tolist() #converting the y-column into a list in the 80% data -Training
test_rooms = testset["Avg. Area Number of Rooms"].tolist() #converting the x-column into a list in the 20% data - Testing
test_price = testset["Price"].tolist() #converting the y-column into a list in the 20% data - Testing
meanroom = sum(test_rooms)/len(test_rooms) # Calculate mean of all x values
meanprice = sum(test_price)/len(test_price) #calculate mean of all y values
train_features = ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Area Population"]

lamda = [0,0.001,0.01,0.1,1,10,100,1000, 10000,100000,1000000, 10 ** 7, 10 ** 8, 10 ** 9] #parameters for lasso and ridge 
mseridge ={}
lasmse = {}
lascofs = {}
ridgecofs = {}


def standardscale(train, valid): #z-score normalization
    mean = train.mean(axis=0)
    std = train.std(axis=0)

    trainscaled = (train - mean)/std
    validscaled = (valid - mean)/std

    return trainscaled, validscaled
    

def outliers(train, valid):
    q1 = train.quantile(0.25)
    q3 = train.quantile(0.75)

    Iqr = q3 - q1
    upperbound = q3 + (1.5 * Iqr)
    lowerbound = q1 - (1.5 * Iqr)

    train = train.clip(lowerbound, upperbound)
    valid = valid.clip(lowerbound,upperbound)
    

def trainmodel(rooms,meanroom,price,meanprice): #called Ordinary Least Squares (OLS)
    sumsone = 0
    sumstwo = 0
    for i in range(0,len(rooms)):
        sumsone = sumsone + (rooms[i] - meanroom)* (price[i] - meanprice) #Mathemathical calculations sum of (xi - xmean)(yi-ymean)
        sumstwo = sumstwo + (rooms[i] - meanroom) ** 2 #Mathemathical calculations sum of xi - xmean)^2
    m = sumsone/sumstwo 
    c = meanprice - (m * meanroom)
    return m, c

def testmodel(test_price, m, c,test_rooms):
    n = len(test_price)
    testsum = 0 #accumulator for sum of (yi - ymean)^2, to calculate MSE
    testsumMAE = 0
    sstot = 0
    meanlocalprice = sum(test_price) / n
    for i in range(0,len(test_price)): #going through all values of 
        testsum = testsum + (test_price[i] - ((m * test_rooms[i]) + c)) ** 2 #finding sum of all model output value for all x values in testset
        testsumMAE = testsumMAE + abs(test_price[i] - (((m * (test_rooms[i])) + c)))
        sstot = sstot + (test_price[i] - meanlocalprice) ** 2
 
    mse = (1/n) * testsum # calculating MSE
    mae = (1/n) * testsumMAE
    rmse = mse ** (1/2) #calculating RMSE/ squarerooting MSE
    r2 = 1 - (testsum/sstot)

    return mse, rmse, mae, r2

def Multipleregression(x,y): # implementation of Normal Equation
    n_rows = x.shape[0]
    ones = np.ones((n_rows, 1))
    X= np.concatenate((ones, x), axis=1)
    Y = y
    Xt = np.transpose(X)

    bhat = ((np.linalg.inv((Xt @ X))) @ Xt) @ Y

    
    return bhat

def testmultiple(b, testrooms, testprice):
    n_rows = testrooms.shape[0]
    ones = np.ones((n_rows, 1))
    testrooms = np.concatenate((ones, testrooms), axis=1)
    y_pred = (testrooms @ b).tolist()
    n = len(testprice)
    meany = sum(testprice) / n
    syi_yi = 0
    syi_yi2 = 0
    rstotal = 0
    for i in range(n):
        syi_yi = syi_yi + abs(testprice[i] - y_pred[i])
        syi_yi2 = syi_yi2 + ((testprice[i] - y_pred[i]) ** 2)
        rstotal = rstotal + ((testprice[i] - meany) ** 2)

    
    mae = (1/n) * syi_yi
    mse = (1/n) * syi_yi2
    rmse = mse ** 0.5
    r2 = 1 - ((syi_yi2)/(rstotal))

    return mae, mse, rmse,r2
#lasso
def lasso(cofs,test_features, actual_price, lamda):
    coefficents = cofs.copy()
    X_raw = test_features
    ones = np.ones((X_raw.shape[0], 1))
    X_with_intercept = np.concatenate((ones, X_raw), axis=1)
    
    for i in range(11): #coeffients lens
        residual = actual_price - (X_with_intercept @ coefficents)
        presidual = residual + (X_with_intercept[:, i] * coefficents[i]) # adding residual with ( 1 column * 1 coeffient) undo trick 

        p_i = np.dot(X_with_intercept[:,i], presidual)
        z_i = np.sum(X_with_intercept[:, i]**2) #z_i is the "denominator" (the squared length of the column)
        if i == 0:
            coefficents[i] = p_i / z_i
        else:
            if p_i > (lamda):
                p_i = p_i - (lamda)
            elif p_i < (-1 * (lamda)):
                p_i = p_i + (lamda)
            else:
                p_i = 0
            coefficents[i] = p_i / z_i

        
    return coefficents

def ridgeregression(lamda,x,y):
    n_rows = x.shape[0]
    ones = np.ones((n_rows, 1))
    X_mat= np.concatenate((ones, x), axis=1)
    Y_mat = y
    
    n_cols = X_mat.shape[1]
    Xt = np.transpose(X_mat)
    I = np.eye(n_cols)
    I[0, 0] = 0
    calc = Xt @ X_mat + (lamda * I)
    bhat = np.linalg.inv(calc) @ Xt @ Y_mat
    return bhat

#linear
sumrmse = 0
summae = 0
sumr2 = 0
#multiple
summmae = 0
summrmse = 0
summr2 = 0
#quadratic
sumqmae = 0
sumqrmse = 0
sumqr2  = 0

for i in range(k):
    valid = folds[i].copy()
    train = pd.concat([folds[noti] for noti in range(k) if noti != i]).copy()
    
    for i in train_features:
        outliers(train[i],valid[i])
        train[i], valid[i] = standardscale(train[i],valid[i]) # standardizing all data
        
    vrooms = train["Avg. Area Number of Rooms"].tolist()
    vprice= train["Price"].tolist()
    vmeanroom = sum(vrooms)/len(vrooms)
    vmeanprice = sum(vprice)/len(vprice)



    
    vtest_rooms = valid["Avg. Area Number of Rooms"].tolist()
    vtest_price = valid["Price"].tolist()
    vtest_pricemean = sum(vtest_price)/len(vtest_price)

    vm, vc = trainmodel(vrooms,vmeanroom,vprice,vmeanprice)
    vmse, vrmse, vmae, vr2 = testmodel(vtest_price, vm, vc, vtest_rooms)
    sumrmse = sumrmse + vrmse
    summae = summae + vmae
    sumr2 = sumr2 + vr2

    X = train[train_features].to_numpy()
    Y = train["Price"].to_numpy()
    Xtest = valid[train_features].to_numpy()
    
    
    b = Multipleregression(X,Y)
    mmae, mmse, mrmse, mr2 = testmultiple(b, Xtest, vtest_price)
    summmae += mmae
    summrmse += mrmse
    summr2  += mr2
    #standardising after squaring terms
    xpoly = np.hstack((X, X**2))
    xpoly_mean = xpoly.mean(axis=0)
    xpoly_std = xpoly.std(axis=0)
    xpoly = (xpoly - xpoly_mean) / xpoly_std
    
    Xpolytest = np.hstack((Xtest, Xtest ** 2))
    xpoly_mean = Xpolytest.mean(axis=0)
    xpoly_std = Xpolytest.std(axis=0)
    xpolytest = (Xpolytest - xpoly_mean) / xpoly_std

    
    polyb = Multipleregression(xpoly,Y)
    qmae, qmse,qrmse,qr2 = testmultiple(polyb, Xpolytest, vtest_price)
    sumqmae += qmae
    sumqrmse += qrmse
    sumqr2  += qr2


    for l in lamda: # ridge and lasso regression being varied through lambda values
        cofs = np.zeros(11)
    
        rideg = ridgeregression(l,xpoly,Y)
        _,ridgemse,Rmseridge,_ = testmultiple(rideg,Xpolytest,vtest_price)
        mseridge[l] = ridgemse
        ridgecofs[l] = rideg[1:]
        while True:
            las = lasso(cofs, xpoly, Y, l)

            if np.linalg.norm(cofs - las) < 0.000001: 
                cofs = las
                
                break
            else:
                cofs = las
        
        _,lasomse,lasrmse,_ = testmultiple(las,Xpolytest,vtest_price)
        #data to plot coefficent magneitude
        lasmse[l] = lasomse #saving MSE for graphs
        lascofs[l] = cofs[1:] # saving cofficients for graph
        

print("Ridge mSE:",mseridge)
print("Lasso MSE",lasmse)
print("COFS:", lascofs)

meanvrmse = sumrmse/5
meanmae = summae/5
r2 = sumr2/5



print("LINEAR REGRESSION")
print("RMSE:",meanvrmse)
print("MAE", meanmae)
print("R^2", r2)

meanmae = summmae/5
meanmrmse = summrmse/5
meanmr2 = summr2/5
radjusted = 1 - (((1-meanmr2) * (len(vtest_price) - 1))/(len(vtest_price) - 5 - 1))

print("MULTIPLE LINEAR REGRESSION")
print("RMSE:", meanmrmse)
print("MAE", meanmae)
print("R^2",meanmr2)
print("Adjusted R^2:", radjusted)

print("Quadratic Regression")

meanqmae = sumqmae/5
meanqrmse = sumqrmse/5
meanqr2 = sumqr2/5
qradjusted = 1 - (((1-meanqr2) * (len(vtest_price) - 1))/(len(vtest_price) - 10 - 1))

print("RMSE:", meanqrmse)
print("MAEL ", meanqmae)
print("R^2", meanqr2)
print("Adjusted R^2", qradjusted)












    
m, c = trainmodel(rooms,meanroom,price,meanprice)

#Graphical representation
x = df["Avg. Area Number of Rooms"].tolist()
y = df["Price"].tolist()
eq = f"y = {m}x + {c}"
eqyvalues = []
for xvalues in x:
    eqyvalues.append((m * xvalues) + c)


plt.title("House Price Against Avg total Rooms")
plt.xlabel('Avg Total Rooms', color="#1C2833")
plt.ylabel('House Price($x10^6)', color="#1C2833")
plt.legend(loc='upper left')
plt.xlim(0, max(x)) # Set x-axis
plt.ylim(0, max(y)+(0.1 * max(y)))
plt.plot(x,y, "x")
plt.plot(x,eqyvalues, label=eq)
plt.legend(loc='upper left')
plt.grid()
plt.show()

xr = np.array(lamda)
yr = np.array(list(mseridge.values()))
yl = np.array(list(mseridge.values()))

fig, ax = plt.subplots(2)
ax[0].plot(xr, yl, marker='o', color='green')
ax[0].set_xscale('log')
ax[0].set_title("Lasso: Lambda vs MSE")
ax[0].set_xlabel("Lambda")
ax[0].set_ylabel("Mean Squared Error")
ax[0].grid(True, which="both", alpha=0.3)

ax[1].plot(xr,yr)
ax[1].set_title("Ridge Regression Error vs. Regularization Strength")
ax[1].set_xscale('log')
ax[1].set_xlabel("Lambda")
ax[1].set_ylabel("MSE")
ax[1].axvline(x=10, color='r', linestyle='--')



plt.tight_layout()
plt.show()


def plotvar(dict):
    xpoints = []
    ypoints = []
    for keys in dict.keys():
        xpoints.append(keys)
        ypoints.append(dict[keys])
        
    plt.plot(xpoints,ypoints)

def plotcofs(lamda,diccofs):
    xpoints = []
    ypoints = []
    for lamdas in lamda:
        for y in diccofs[lamdas]:
            xpoints.append(lamdas)
            ypoints.append(y)


    return xpoints, ypoints



#plotting coeffient lambda
plt.subplot(2,1,1)
lasx, lasy = plotcofs(lamda,lascofs)
for i in range(10):
    plt.plot(lasx[i::10], lasy[i::10])
plt.title("Coefficent-Lambda Lasso")
plt.xscale('log') 
plt.xlabel("Lambda/Penalty")
plt.ylabel("Coefficent")

plt.subplot(2,1,2)
#print("\n",ridgecofs)
ridx, ridy = plotcofs(lamda,ridgecofs)
for i in range(10):
    plt.plot(ridx[i::10], ridy[i::10])
plt.title("Coefficent-Lambda Ridge")
plt.xscale('log') 
plt.xlabel("Lambda/Penalty")
plt.ylabel("Coefficent")
plt.show()




