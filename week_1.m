#Activity from the machine learning course from Stanford University. The code was written on Matlab as one of the assignments.

clear ; close all; clc

## Implementation of a linear regression model for 2D data

##===================Loading the data
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);

##===================Plotting the data
figure; 
plot(X,y, 'rx', 'MarkerSize', 10)
xlabel('Population')
ylabel('Profit')

##===================Cost and Gradient descent
X = [ones(m, 1), data(:,1)]; 
theta = zeros(2, 1); 

iterations = 1500;
alpha = 0.01;

function J = computeCost(X, y, theta)
  m = length(y)
  h_theta = theta(1)+theta(2)*X(:,2);
  J = 1/(2*m)*sum((h_theta-y).^2);
end
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y)
  J_history = zeros(num_iters, 1);
  for iter = 1:num_iters
    J_history(iter) = computeCost(X, y, theta, m);
    h_theta = theta(1)+theta(2)*X(:,2);
    theta_0 = theta(1) - (alpha/m)*sum((h_theta-y).*X(:, 1));
    theta_1 = theta(2) - (alpha/m)*sum((h_theta-y).*X(: ,2));
    theta = [theta_0; theta_1];
  end
end
##==============================================================================
## Implementation of a linear regression model for 2D
clear ; close all; clc

##===================Loading the data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);

##===================Feature normalization
function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  [m, n] = size(X);
  mu = zeros(1, n);
  sigma = zeros(1, n);

  mu = mean(X_norm);
  sigma = std(X_norm);
  for i = 1:n
    X_norm(:,i) = (X_norm(:,i) - mu(i))./sigma(i);
  end
end

##===================Cost and Gradient descent for multivariables
X = [ones(m, 1) X];
alpha = 0.08;
num_iters = 50;
theta = zeros(3, 1);

function J = computeCostMulti(X, y, theta)
  [m, n] = size(X);
  h_theta = zeros(m, 1);
  for i = 1:n
    h_theta = h_theta+theta(i)*X(i);
  endfor
  J = 1/(2*m)*sum((h_theta-y).^2);
end
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  [m, n] = size(X);
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters
    J_history(iter) = computeCostMulti(X, y, theta);
    h_theta = zeros();
    for i = 1: n
      h_theta = h_theta+theta(i)*X(:, i);
    end
    for j = 1: n
      theta(j) = theta(j)- (alpha/m)*sum((h_theta-y).*X(:, j));
    end
  end
end

##===================Normal Equation
function [theta] = normalEqn(X, y)
theta = zeros(size(X, 2), 1);
theta = inv(X'*X)*X'*y
end
