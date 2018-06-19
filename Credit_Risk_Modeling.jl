
# import required packages 
using DataFrames
using FreqTables
using Plots, StatPlots
using DecisionTree
using StatsModels
using MLDataUtils

#Set the backend as matplotlib.pyplot
pyplot()

# Read Dataset
data = readtable("C:/Users/veer/Desktop/Projects/Julia_In_Banking/Data/loan_data.csv")

# Create 02 categories for loan status
data[:loan_status_new] = ifelse.(data[:loan_status] .== "Fully Paid",0,1)
delete!(data,:loan_status)

rename!(data, :loan_status_new, :loan_status)

# Delete Loan ID and Customer ID columns
delete!(data,:loan_id)
delete!(data,:member_id)

# Explore Loan Amount
freqtable(data[:loan_status])

describe(data[:annual_inc])

# Plot Histogram for Applicant Income
Plots.histogram(data[:annual_inc], bins = 1000, xlabel = "annual_inc", labels = "Frequency")

eltypes(data)

freqtable(data[:home_ownership])

# Partition Data into Train and Test datasets
function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

train,test = partitionTrainTest(data, 0.7) # 70% train

# Missing value imputation-- train
# Replace missing loan amount with median of loan amount
showcols(train)
train[isna.(train[:emp_length]),:emp_length] = floor(median(dropna(train[:emp_length])))

# Missing value imputation-- test
# Replace missing loan amount with median of loan amount
showcols(test)
test[isna.(test[:emp_length]),:emp_length] = floor(median(dropna(test[:emp_length])))

# Build Logistic Regression Model
# Pkg.add("GLM")
# Load GLM Library
using GLM

# Build Logistic Regression Model
model_LG = glm(@formula(loan_status ~ annual_inc + loan_amnt + int_rate + cibil_score + inq_last_6mths
        + age + emp_length + Rent + Mortgage + _36_months + married + single + job_fixed + job_freelance
        + job_parttime), train, Binomial(), ProbitLink())

# Confusion Matrix
confint(model_LG)

train

# Build Decision Tree Model
y_train = train[:,:loan_status]
X_train = train[:,[1,2,3,4,5,6,7,8,9,10,11]]
y_test = test[:,:loan_status]
X_test = test[:,[1,2,3,4,5,6,7,8,9,10,11]]

# Undersampling the training dataset
X_train, y_train = undersample((X_train, y_train), shuffle = true)

# Build Tree
# Train full-tree classifier
model_DT = build_tree(y_train, X_train)

# Prune tree: merge leaves having >= 90% combined purity (default: 100%)
model_DT = prune_tree(model, 0.9)

# print the tree, to a depth of 5 nodes
print_tree(model_DT, 5)

# test on the Test dataset
y_pred1 = apply_tree(model_DT, X_test1)

freqtable(y_test1,y_pred1)

using RCall

saveRDS("C:/Users/veer/Desktop/Projects/Credit_Risk_Modeling/inal/JuliaModelCreditLog.rds",model_Log)

saveRDS("C:/Users/veer/Desktop/Projects/Credit_Risk_Modeling/inal/JuliaModelCreditDT.rds",model_DT)
