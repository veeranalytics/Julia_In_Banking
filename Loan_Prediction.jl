
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

# Label Encoding-- Home Ownership
data[:Rent] = ifelse.(data[:home_ownership] .== "RENT",1,0)
data[:Mortgage] = ifelse.(data[:home_ownership] .== "MORTGAGE",1,0) # Zero values for both the home ownership type show OWN type
delete!(data,:home_ownership)

# Label Encoding-- term
data[:_36_months] = ifelse.(data[:term] .== "36 months",1,0)
delete!(data,:term)

# Label Encoding-- marital
data[:married] = ifelse.(data[:marital] .== "married",1,0)
data[:single] = ifelse.(data[:marital] .== "single",1,0)
delete!(data,:marital)

# Label Encoding-- marital
data[:job_fixed] = ifelse.(data[:job] .== "fixed",1,0)
data[:job_freelance] = ifelse.(data[:job] .== "freelance",1,0)
data[:job_parttime] = ifelse.(data[:job] .== "partime",1,0)
delete!(data,:job)

# Partition Data into Train and Test datasets
function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

train,test = partitionTrainTest(data, 0.7) # 70% train

# Missing value imputation- train dataset
# Replace missing loan amount with median of loan amount
showcols(train)
train[isna.(train[:emp_length]),:emp_length] = floor(median(dropna(train[:emp_length])))

# Missing value imputation-- test dataset
# Replace missing loan amount with median of loan amount
showcols(test)
test[isna.(test[:emp_length]),:emp_length] = floor(median(dropna(test[:emp_length])))

showcols(train)

showcols(test)

y_train = train[:,:loan_status]
X_train = train[:,[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]]
y_test = test[:,:loan_status]
X_test = test[:,[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]]

# Undersampling the training dataset
X_train, y_train = undersample((X_train, y_train), shuffle = true)

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
    StandardScalar(Array(Real,0),Array(Real,0))
end

# Compute mean and standard deviation of each column
function fit_std_scalar!(std_scalar::StandardScalar,X::Matrix{Real})
    n_rows, n_cols = size(X)
    std_scalar.std = zeros(n_cols)
    std_scalar.mean = zeros(n_cols)
    # for loops are fast again!
    for i = 1:n_cols
        std_scalar.mean[i] = mean(X[:,i])
        std_scalar.std[i] = std(X[:,i])
    end
end

function transform(std_scalar::StandardScalar,X::Matrix{Real})
    (X .- std_scalar.mean') ./ std_scalar.std' # broadcasting
end

# fit and transform in one function
function fit_transform!(std_scalar::StandardScalar,X::Matrix{Real})
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

# Pkg.add("RCall.jl")
using RCall

saveRDS("C:/Users/veer/Desktop/Projects/Credit_Risk_Modeling/Final/JuliaModel.rds", model)
