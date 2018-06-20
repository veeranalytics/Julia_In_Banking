
# import required packages 
using DataFrames

# Function for Confusion Matrix
function confusion_matrix(y_true::Array{Int64,1},y_pred::Array{Int64,1})
    # Generate confusion matrix
    classes = sort(unique([unique(y_true),unique(y_pred)]))
    cm = zeros(Int64,length(classes),length(classes))

    for i in 1:length(y_test)
        # translate label to index
        true_class = findfirst(classes,y_test[i])
        pred_class = findfirst(classes,y_pred[i])
        # pred class is the row, true class is the column
        cm[pred_class,true_class] += 1
    end
    cm
end

# Example: confusion_matrix(y_test,y_pred)

# Partition Data into Train and Test datasets
function partitionTrainTest(data, at = 0.7)
    n = nrow(data)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

# Example: train,test = partitionTrainTest(data, 0.7) # 70% train

# Build a standard scaler function
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
    n_rows, n_cols = size(X_test)
    std_scalar.std = zeros(n_cols)
    std_scalar.mean = zeros(n_cols)
    # for loops are fast again!
    for i = 1:n_cols
        std_scalar.mean[i] = mean(X[:,i])
        std_scalar.std[i] = std(X[:,i])
    end
end

function transform(std_scalar::StandardScalar,X::Matrix{Real})
    (X .- std_scalar.mean') ./ std_scalar.std' # broadcasting fu
end

# fit and transform in one function
function fit_transform!(std_scalar::StandardScalar,X::Matrix{Real})
    fit_std_scalar!(std_scalar,X)
    transform(std_scalar,X)
end

# Examples:
# Perform Standard Scaling for all X variables
# std_scalar = StandardScalar()
# X_train = fit_transform!(std_scalar,X_train)
# X_test = transform(std_scalar,X_test)

# Logistic Regression
function predict(data, model_params) 
	1.0 ./ (1.0 + exp(-data * model_params[:omega]  - model_params[:beta]))
end 

function _mult(a::Array{Float64,1},b::Array{Float64,2})
    result = zeros(length(a))
    both_non_zero_indicator = ((a .!= 0) &amp; (b .!= 0))
    result[both_non_zero_indicator[:]] = a[both_non_zero_indicator[:]] .* b[both_non_zero_indicator]
    return result
end


function goal_function(omega::Array{Float64,2}, beta::Float64, data::Array{Float64,2}, labels::Array{Float64,1})
    f_partial = 1.0 ./ (1.0 + exp(-data * omega  - beta))
    result = -sum(_mult(labels, log(f_partial)) + _mult((1.0 - labels), log(1.0 - f_partial)))
    return result
end

function convergence(omega::Array{Float64,2}, beta::Float64, data::Array{Float64,2}, labels::Array{Float64,1},  prevJ::Float64, epsilon::Float64)
     currJ = goal_function(omega, beta, data, labels)
     return abs(prevJ - currJ) &lt; epsilon
end

function update_params(omega::Array{Float64,2}, beta::Float64,data::Array{Float64,2}, labels::Array{Float64,1}, alpha::Float64)
    partial_derivative = (1.0 ./ (1.0 + exp(-data * omega  - beta)))  - labels
    omega = omega - alpha *  (partial_derivative' * data)'
    beta = beta  - alpha * sum(partial_derivative)
    return omega,beta
end

function logistic_regression(data::Array{Float64,2}, labels::Array{Float64,1}, params::Dict{Symbol,Float64})

    omega = zeros(Float64, size(data,2),1)
    beta = 0.0
    J = Inf
    current_iter = 0
    alpha_step, epsilon, max_iter = params[:alpha], params[:eps], params[:max_iter]

    while !convergence(omega, beta, data, labels, J, epsilon) &amp;&amp; current_iter &lt; max_iter
         J = goal_function(omega, beta, data, labels)
         omega, beta = update_params(omega, beta, data, labels, alpha_step)
         current_iter += 1
    end
    model_params = Dict();
    model_params[:omega] = omega; model_params[:beta] = beta;	
    return model_params
end
