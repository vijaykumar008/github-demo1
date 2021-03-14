Important Commands

#Deafutly iam considering Data as a Dataset(Which is a DataFrame)

---dataset.isnull().sum()# to check any NULLS in given dataset

--sns.countplot(x='Outcome',data=dataset)# It will show No.of zero's and No.of 0ne's in given column(Outcome)
plt.show()

--data.groupby('Outcome').size()# It will show No.of zero's and No.of 0ne's in given column(Outcome)


--dataset.hist(figsize=(1,1))# to view histogram graph of dataset
--sns.distplot(df_train['SalePrice']);## to view histogram graph of dataset

---Corr=Data[Data.columns].corr()#To know correlation between columns
sns.heatmap(Corr,annot=True)# To view correletion matrix in pictorial reresenttion 

----corr = Data.corr()

---dataset.columns#To know column names in dataset

--dataset.SalePrice.skew()#To know skewness value of column(SAleprice)
----dataset.SalePrice.kurt()#To know kurtousis value of column(SAleprice)

----print(dataset.column.unique())#Gives unique values in column
----len(set(dataset.column))#Gives unique values in column

---df=pd.merge(animal,ani_class,how='left',left_on='class_type',right_on='Class_Number')#How to merge two tables

--DataFrame.drop#is used to drop or delete the rows or columns

--dataset.column.value_counts()# gives the count of variables in a column

--drop_duplicates()#Delets duplicate values in dataset

--groupby()# commond is udefull to group two perticular column
----group_df = Data.groupby(pd.Grouper(key=v)).mean()

---train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar() #To group two features
--agg()#Means grouing the terms
            
---dataset.sort_values('columnname') #for sorting

---datset['newcolumn'] = dataset['column1'] / dataset['column2'] # creating a new colmn by adding existing columns

---string.punctuation#it gives output '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

---string.digits  #it gives outut '0123456789'
---df = df.dropna() #Drops the missing values

#Below code used for correlation gives better results with color
--matrix = train.corr()
--f, ax = plt.subplots(figsize=(9, 6))
--sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");

---Data.rename(columns ={'price': 'SalePrice'}, inplace =True)# To rename a column

--sns.distplot(Data.SalePrice, bins=50, kde=False); #To visualize a column in graphical representation

---print("Skewness: %f" % Data.SalePrice.skew()) #To print a value along with the comment
---print("Total number ber of people involve in the test: {}".Data.SalePrice.skew()) #To print a value along with the comment

--Data.SalePrice.skew() #To find skewness of the column
---Data.skew() #To find skewness of whole data

---Data.SalePrice.kurt() #To find kurtousis of the column

---cols = ['SalePrice', 'sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms'] #To draw a pairplot between columns
---sns.pairplot(Data.cols, size = 2.5)

----Data.SalePrice = np.log(dData.SalePrice) #It is a log transformation(it is used for
					Converting a feature into normal distribution)
---Data.corr() #gives correlation between all the features

--accuracy = regressor.score(x_test, y_test) #We can find the accuracy for Test data

#In a Gender column we want to know how many entries for FEMALE, Use the below command
---Data[Data.label == 'female'].shape[0] (or) format(Data[Data.label == 'female'].shape[0])

---X = Data.iloc[:,:-1] #Selecting the Independent/Input Features(If dependent feature is at last column)
---y = Data.iloc[:,-1]  #Selecting the Dependent/Output features(If dependent feature is at last column)

---Data.Gender.value_counts() #It will gives number of Males and Females in the column
---Data.Gender.value_counts().plot.bar() #To plot a graph

---Gender=pd.crosstab(train.Gender,train.Loan_Status) #Displays graph between two features (Gender and Loan_status are two features, this command 
						    will give how many male applicants and female applicants got loan)	
---Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


---train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) #To replace missing value with mode.

---train=pd.get_dummies(train) #This command will generate dummy variables for Categorical variables in total Train dataset.

---np.where(data["score_reading"]>=40, "P", "F") #In Score_reading column it replace value by P if value greater than 40 and F by less than 40

--sns.countplot(x = data["race"])) #It will give the count of different categeries present in "Race" column,
                                  # with graphical representation
                                  
--dfy = df[df.Churn == 'Yes'] #Churn column has yes/no values, this command will divides data which 
                                #has value of yes, it will generates saparate data set
                                  

----a = raw_input().split()# It will divide he input with saces 
                            #example 1 2 3

------b= int(raw_input())/ b = int(input()) # it will takes the integer as raw input ex: 5                            
-----a =raw_input()/input()  #This we can take for string input(we can use any of those)

---Suppose if you want to declare a random integer and that random integer 
   np.of elements inserted into set(or may into list/tuple) use the below command
-----t = int(input())
    a = set(map(int,raw_input().split()))   
    
N=int(input()) #If we want integer as input

N,X = map(int,raw_input().split()) #If we want to give 2 integer values as input with space
n = int(input("enter u r input")) #for default input with comment