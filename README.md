# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Objective
The objective of this project is to build a feed-forward neural network to predict customer churn and analyze how neural networks learn through forward propagation, loss computation, backpropagation, and parameter updates.

## Dataset
This project uses the provided `customer_churn_nn.csv` dataset, from the link: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

### Target Variable
- `churn`
  - 1 = Customer churned
  - 0 = Customer retained

### Dataset Characteristics
- 2,000 rows
- 17 columns
- Highly imbalanced target distribution

### Feature Types
#### Categorical Features
- region
- plan_type
- contract_type
- payment_method

#### Numerical Features
- tenure_months
- monthly_charges_inr
- avg_login_days_per_month
- support_tickets_last_90_days
- payment_delay_days
- monthly_data_usage_gb
- satisfaction_score
- complaint_recency_days
- discount_received_pct
- referrals_count

### Ignored Column
- customer_id (identifier only)



## Methodology

### 1. Dataset Understanding
- Loaded the dataset
- Checked shape and data types
- Verified missing values
- Generated summary statistics
- Visualized target distribution

### 2. Data Preprocessing
- Removed `customer_id`
- Imputed missing values
- One-hot encoded categorical variables
- Standardized numerical variables
- Split data using stratified train-test split

### 3. Neural Network Model
Architecture:
- Input layer
- Hidden Layer 1: 32 neurons, ReLU
- Dropout: 0.2
- Hidden Layer 2: 16 neurons, ReLU
- Dropout: 0.2
- Output Layer: 1 neuron, Sigmoid

Loss Function:
- Binary Crossentropy

Optimizer:
- Adam

### 4. Training
- Used class weights to address class imbalance
- Used early stopping to prevent overfitting
- Trained for up to 100 epochs

### 5. Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 6. Hyperparameter Experiments
Four experiments were conducted by varying:
- Hidden layers
- Number of neurons
- Learning rate
- Batch size
- Activation function

## Results
The best model was selected based on Recall and F1-Score for the churn class.

Since the dataset is highly imbalanced, Recall is particularly important because it measures the percentage of actual churners correctly identified.

## Key Findings
- Class weighting significantly improved churn detection.
- ReLU activation generally performed better than tanh.
- Deeper networks slightly improved performance.
- Early stopping prevented overfitting.

## Final Reflection

### 1.What role do weights and biases play in the model?
Weights determine how strongly each input feature influences a neuron’s output. During training, the model adjusts these weights to minimize prediction error.

Biases are additional learnable parameters that allow neurons to shift the activation function. They help the model fit the data more flexibly.

### 2.Why is an activation function required?
Activation functions introduce non-linearity into the network. Without them, no matter how many layers are added, the model would behave like a simple linear model.

Functions such as ReLU and Sigmoid introduce non-linearity allowing the neural network to learn complex patterns and relationships.

### 3.What happens when learning rate is too high or too low?
- Too high: The optimizer may overshoot the optimal solution and fail to converge, causing unstable training or failure to converge.
- Too low: Training becomes very slow and may get stuck in poor local minima before reaching a good solution. 

A suitable learning rate balances speed and stability.

### 4.Did the model show signs of underfitting or overfitting?
The training and validation curves were used to assess model behavior.

If both training and validation performance are poor, the model is underfitting. If training performance is excellent but validation performance is much worse, the model is overfitting.

In this project, early stopping and dropout were used to reduce overfitting.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt