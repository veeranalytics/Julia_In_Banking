
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
data = readtable("C:/Users/veer/Desktop/Projects/Julia_In_Banking/Data/fraud_full_sample.csv")

# Create 02 categories (numeric) for loan status
data[:fraud_status] = ifelse.(data[:isFraud] .== "Yes",1,0)
delete!(data,:isFraud)

rename!(data, :fraud_status, :isFraud)

# Explore Loan Amount
freqtable(data[:isFraud])

describe(data[:amount])

# Plot Histogram for Applicant Income
Plots.histogram(data[:amount], bins = 1000, xlabel = "Transaction_Amount", labels = "Frequency")

eltypes(data)

freqtable(data[:_type])

# Label Encoding-- Transaction Type
data[:CASH_OUT] = ifelse.(data[:_type] .== "CASH_OUT",1,0) # The value 0 is for Transfer type transaction
delete!(data,:_type)

# Partition Data into Train and Test datasets
function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

train,test = partitionTrainTest(data, 0.7) # 70% train

# Missing value imputation
showcols(train)
# There is no missing value, therefore this step is not required

y_train = train[:,:isFraud]
X_train = train[:,[1,2,3,4,5,6,8]]
y_test = test[:,:isFraud]
X_test = test[:,[1,2,3,4,5,6,8]]

# Convert all the datasets in to arrays
# Convert data types for all the columns in training dataset to float type
for c = eachcol(X_train)
  if eltype(c[2]) <: Integer
    X_train[c[1]] = X_train[c[1]] .* 1.0
  end
end

for c = eachcol(X_test)
  if eltype(c[2]) <: Integer
    X_test[c[1]] = X_test[c[1]] .* 1.0
  end
end

X_train1 = convert(Array, X_train)
X_test1 = convert(Array, X_test)
y_train1 = convert(Array, y_train)
y_test1 = convert(Array, y_test)

# Build a standard scaler
type StandardScalar
    mean::Vector{Float64}
    std::Vector{Float64}
end

# Helper function to initialize an empty scalar
function StandardScalar()
    StandardScalar(Array(Float64,0),Array(Float64,0))
end

# Compute mean and standard deviation of each column
function fit_std_scalar!(std_scalar::StandardScalar,X::Matrix{Float64})
    n_rows, n_cols = size(X)
    std_scalar.std = zeros(n_cols)
    std_scalar.mean = zeros(n_cols)
    # for loops are fast again!
    for i = 1:n_cols
        std_scalar.mean[i] = mean(X[:,i])
        std_scalar.std[i] = std(X[:,i])
    end
end

function transform(std_scalar::StandardScalar,X::Matrix{Float64})
    (X .- std_scalar.mean') ./ std_scalar.std' # broadcasting
end

# fit and transform in one function
function fit_transform!(std_scalar::StandardScalar,X::Matrix{Float64})
    fit_std_scalar!(std_scalar,X)
    transform(std_scalar,X)
end

# Perform Standard Scaling for all X variables
std_scalar = StandardScalar()

X_train1 = fit_transform!(std_scalar,X_train1)
X_test1 = transform(std_scalar,X_test1)

labels= y_train1
features= X_train1

# train random forest classifier
# using 2 random features, 10 trees, 0.5 portion of samples per tree (optional), and a maximum tree depth of 6 (optional)
model = build_forest(labels, features, 2, 100, 0.5, 10)

# test on the Test dataset
y_pred1 = apply_forest(model, X_test1)

freqtable(y_test1,y_pred1)

accuracy = (2337 + 2468)/length(y_test1)

using RCall

saveRDS("C:/Users/veer/Desktop/Projects/Credit_Risk_Modeling/inal/JuliaModelCreditLog.rds",model_Log)

saveRDS("C:/Users/veer/Desktop/Projects/Credit_Risk_Modeling/inal/JuliaModelCreditDT.rds",model_DT)
