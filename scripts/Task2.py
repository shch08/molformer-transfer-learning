# Import dependencies
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset,  TensorDataset, ConcatDataset
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import anderson
import seaborn as sns

# macros
BATCH_SIZE = 16
RECURSION_DEPTH = 10
DAMPING = 0.01

# MODEL_PATH is a stored pretrained model from task 1, this will loaded for all calculation further
MODEL_PATH = "/home/neuronet_team130/nnti_project/molformer_mlm"

# Hugging face model name and dataset path
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"


# This function reshapes the flat vector to match the parameter shape
def unflatten_vector(vector, params):
    # select the device based on availability
    # This is important to make sure all the calculation necessary vectors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pointer = 0
    
    # Initialize unflatten vector
    unflattened = []
    for param in params:
        num_params = param.numel()
        # Extract corresponding segment from flattened vector using slicing method and reshape it
        unflattened.append(vector[pointer:pointer+num_params].view(param.shape).to(device))
        pointer += num_params
    return unflattened

# This function calculates the Hessian-vector product (HVP)
def compute_hvp(loss, params, v):
    # select the device based on availability
    # This is important to make sure all the calculation necessary vectors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move all necessary input tensors to device
    v = v.to(device)
    loss = loss.to(device)

    # Calculate gradients using autograd
    grad_params = autograd.grad(loss, params, create_graph=True, retain_graph=True)

    # Reshapes the vector v to match the shapes of individual parameters and load it to device
    v_unflattened = unflatten_vector(v, params) 
    v_unflattened = [v_i.to(device) for v_i in v_unflattened]

    # compute Hessian-vector product (HVP)
    hvp = autograd.grad(grad_params, params, grad_outputs=v_unflattened, retain_graph=True)

    # replaces any None values in the HVP with zeros
    hvp = [h.to(device) if h is not None else torch.zeros_like(p, device=device) for h, p in zip(hvp, params)]

    # Faltten each tensor and concatenate them into a single 1D tensor
    return torch.cat([g.contiguous().view(-1) for g in hvp])

# This function Computes an approximation of the inverse Hessian-vector product (H^-1 v) using LiSSA
def lissa_approximation(loss, params, v, damping=0.01, recursion_depth=10):
    # select the device based on availability
    # This is important to make sure all the calculation necessary vectors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move all necessary input tensors to device
    v = v.to(device)
    ihvp = v.clone().to(device)

    # Run through the loop for recursion_depth
    for _ in range(recursion_depth):

        # Compute HVP
        hvp = compute_hvp(loss, params, ihvp)

        # Check if there are any NaN or Infinity values , replace them with zero
        if torch.isnan(hvp).any() or torch.isinf(hvp).any():
            print(f"Warning: NaN or Inf detected in HVP at iteration {_}, replacing with zeros")
            hvp = torch.where(torch.isnan(hvp) | torch.isinf(hvp), 
                             torch.zeros_like(hvp), 
                             hvp)

        # Values are clamped between -10 and 10 to prevent extreme values
        hvp = torch.clamp(hvp, -10, 10)

        # Move hvp to device           
        hvp = hvp.to(device)

        # Calculate the Inverse HVP
        ihvp = v + (1 - damping) * ihvp - hvp

        # Scale down the approximation to prevent explosion of values
        ihvp_norm = torch.norm(ihvp)
        if ihvp_norm > 10:
            ihvp = ihvp * (10 / ihvp_norm)

        # function stops early if numerical stability issues arise
        if torch.isnan(ihvp).any() or torch.isinf(ihvp).any():
            print("Warning: NaN or Inf detected in LiSSA, stopping recursion.")
            break
    return ihvp

# This class converts the raw data to tokenization of SMILES strings and their labels
class SMILESDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.is_pandas = hasattr(dataset, 'iloc')

    # Length of the dataset    
    def __len__(self):
        return len(self.dataset)

    # Convert raw data to tokenization of SMILES strings and their labels 
    def __getitem__(self, idx):
        # If dataset is a pandas frame
        if self.is_pandas:
            row = self.dataset.iloc[idx]
            smiles = row["SMILES"]
            label = torch.tensor(row["Label"], dtype=torch.float)
        # If dataset is from huggingface
        else:
            row = self.dataset[idx]
            smiles = row["SMILES"]
            label = torch.tensor(row["label"], dtype=torch.float)
        
        # Uses the tokenizer to convert the SMILES string to token ID
        inputs = self.tokenizer(
            smiles, padding="max_length", truncation=True, return_tensors="pt", max_length=128, return_attention_mask=True
        )
        return inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0), label

# This function computes the influence of each external dataset point on a test point
def compute_influence(model, test_dataset, external_dataset, loss_fn, recursion_depth=10, damping=0.01, batch_size=32):
    # select the device based on availability
    # This is important to make sure all the calculation necessary vectors are on the same device
    device = next(model.parameters()).device

    # Creates PyTorch DataLoaders for both external and test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    external_loader = DataLoader(external_dataset, batch_size=1, shuffle=False)
    
    # Compute test gradients
    model.zero_grad()
    total_test_grad = None
    num_test_samples = 0

    # For each data point in test set calculate gradient
    for inputs, attention_mask, targets in test_loader:
        inputs, attention_mask, targets = inputs.to(device), attention_mask.to(device), targets.to(device)

        # Do the forward pass and calculate the loss function
        model_output = model(inputs, attention_mask)
        outputs = model_output.logits.mean(dim=1)
        outputs = outputs.mean(dim=1, keepdim=True)
        targets = targets.view(-1, 1)
        loss = loss_fn(outputs, targets)

        # backward pass
        loss.backward(retain_graph=True)

        # extract the gradients
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
                grads.append(p.grad.detach().cpu().view(-1))
        flat_grad = torch.cat(grads)

        # Check if there are any NaN or Infinity values , replace them with zero
        if torch.isnan(flat_grad).any() or torch.isinf(flat_grad).any():
            print("Warning: NaN or Inf detected in gradients, replacing with zeros")
            flat_grad = torch.where(torch.isnan(flat_grad) | torch.isinf(flat_grad), 
                                   torch.zeros_like(flat_grad), 
                                   flat_grad)
            
        # Normalizes gradient vector by its norm to ensure stable scaling
        norm = torch.norm(flat_grad)
        if norm > 0:
            flat_grad = flat_grad / norm

        # assemble all test gradients across the batches
        if total_test_grad is None:
            total_test_grad = flat_grad * inputs.size(0)
        else:
            total_test_grad += flat_grad * inputs.size(0)
        num_test_samples += inputs.size(0)
    avg_test_grad = total_test_grad / num_test_samples

    # Normalizes the test gradient vector
    grad_norm = torch.norm(avg_test_grad)
    if grad_norm > 0:
        avg_test_grad = avg_test_grad / grad_norm
    
    # Compute inverse Hessian-vector product
    inv_hvp = lissa_approximation(loss, list(model.parameters()), avg_test_grad, damping, recursion_depth)

    # Check if there are any NaN or Infinity values , replace them with zero
    if torch.isnan(inv_hvp).any() or torch.isinf(inv_hvp).any():
        print("Warning: NaN or Inf detected in inverse HVP, replacing with zeros")
        inv_hvp = torch.where(torch.isnan(inv_hvp) | torch.isinf(inv_hvp), 
                             torch.zeros_like(inv_hvp), 
                             inv_hvp)

    # Normalizes inverse HVP
    ihvp_norm = torch.norm(inv_hvp)
    if ihvp_norm > 0:
        inv_hvp = inv_hvp / ihvp_norm

    # Move inverse HVP to the device               
    inv_hvp = inv_hvp.to(device)
    
    # Compute influence of each external data point
    influences = {}
    # For each data point in external dataset
    for i, (inputs, attention_mask, targets) in enumerate(external_loader):
        inputs, attention_mask, targets = inputs.to(device), attention_mask.to(device), targets.to(device)

        # Computes loss for external dataset point and backpropagation
        model.zero_grad()
        model_output = model(inputs, attention_mask)
        outputs = model_output.logits.mean(dim=1)
        outputs = outputs.mean(dim=1, keepdim=True)
        targets = targets.view(-1, 1)
        loss = loss_fn(outputs, targets)
        loss.backward(retain_graph=True)

        # extract the gradients
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
                grads.append(p.grad.detach().cpu().view(-1))
        flat_grad = torch.cat(grads)

        # Check if there are any NaN or Infinity values , replace them with zero
        if torch.isnan(flat_grad).any() or torch.isinf(flat_grad).any():
            flat_grad = torch.where(torch.isnan(flat_grad) | torch.isinf(flat_grad), 
                                   torch.zeros_like(flat_grad), 
                                   flat_grad)
        
        # Normalize gradient vector
        norm = torch.norm(flat_grad)
        if norm > 0:
            flat_grad = flat_grad / norm
        flat_grad = flat_grad.to(device)

        # Compute dot product of the inverse HVP and external data point gradient, also scale it
        influence = -torch.dot(inv_hvp, flat_grad).item()
        influence = influence * 100 
        influences[i] = influence
    
    return influences

# This function analyses the influence score and make statistical inferences
def analyze_influence_scores(results_df):
    # extract influence scores
    influence_scores = results_df['influence_score'].values

    # Plot the histogram
    plt.hist(influence_scores, bins=30, density=True, alpha=0.6, color='b')
    plt.xlabel("Influence Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Influence Scores")
    plt.show()
    plt.savefig("Historgram.png")

    # Anderson-Darling Test to check Normal distribution
    result = anderson(influence_scores, dist='norm')
    print(f"Anderson-Darling Test: Test Statistic = {result.statistic}")
    print("Critical Values:", result.critical_values)
    print("Significance Levels:", result.significance_level)
    if result.statistic < result.critical_values[2]: 
        print("The influence scores likely follow a normal distribution.")
    else:
        print("The influence scores do not follow a normal distribution.")

    # Influence score distribbution with selection threshold
    threshold = np.percentile(influence_scores, 33) 
    sns.histplot(influence_scores, bins=30, kde=True, color='blue', label="All Influence Scores")
    plt.axvline(threshold, color='red', linestyle='dashed', label="Selection Threshold")
    plt.legend()
    plt.xlabel("Influence Scores")
    plt.ylabel("Frequency")
    plt.title("Influence Score Distribution with Selection Threshold")
    plt.show()
    plt.savefig("Sns.png")


# This function saves influence scores and the corresponding input data points to a CSV file
def save_influence_results(influence_scores, external_dataset, output_file="influence_results.csv"):
    # Create a list to store results
    results = []
    
    # Iterate through influence scores
    for idx, score in influence_scores.items():
        # Extract the data point information
        row = external_dataset.dataset.iloc[idx]
        row_dict = row.to_dict()
        row_dict["influence_score"] = score
        results.append(row_dict)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Sort by influence score (lower to higher values)
    results_df = results_df.sort_values(by="influence_score", ascending=True)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Saved influence results to {output_file}")
    
    # Choose top external data points 
    # As influence score is a negative dot product, negative influence score are important point
    # hence to choose selected points based on the threshold calculated.
    # Based on the analysis of the distribution of the influence scores we can see lower end of the tail is 

    analyze_influence_scores(results_df)
    influence_scores_ordered = results_df['influence_score'].values
    threshold = np.percentile(influence_scores_ordered, 33) 
    selected_points = results_df[results_df['influence_score'] <= threshold]
    top_smiles_labels = selected_points[['SMILES', 'Label']].reset_index(drop=True)

    return results_df, top_smiles_labels

# Reused froom task 1 for consistency 
# This function attacjes the regrssion head at the end of the model as this is a regression task
class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, curr_model):
        #to behave like a PyTorch-model, we super
        super(MoLFormerWithRegressionHead, self).__init__()
        self.model = curr_model
        #a single linear layer we use for the regression head
        #768 is common for transformers
        self.regression_head = nn.Linear(768, 1)
        
    def forward(self, input, attention_masks):
        #first get results after final hidden layer of the MolFormer-model
        outputs_molformer = self.model(input_ids=input, attention_mask=attention_masks)
        pooled_output = outputs_molformer.last_hidden_state[:, 0, :]
        
        #then additionally run our defined regression head for final model output
        regression_output = self.regression_head(pooled_output)
        
        return regression_output.squeeze(-1)

# Reused from task 1 for consistency 
# This function is used to train the model
def train_model(regression_model, train_dataloader, lr=2e-5, epochs=5):
    #standart optimizer and loss for transformers and regression
    optimizer = torch.optim.AdamW(regression_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    regression_model.train()
    all_losses = []

    for epoch in range(epochs):
        #additionally compute avg loss for each epoch for evaluation
        total_loss = 0
        count_batch = 0
        
        #for each epoch run trough all trainingssamples
        for input, attention_mask, labels in train_dataloader:
            #for each trainingssample run forward and backpropagate trough our model
            input = input.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            #run forward
            outputs = regression_model(input = input, attention_masks=attention_mask)
            #compute loss and backpropagate
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            #sum up losses
            total_loss += loss.item()
            count_batch += 1
        
        #compute and store avg loss for every epoch
        avg_loss = total_loss / count_batch
        all_losses.append(avg_loss)
        
        print("Training-Epoch Nr." + str(epoch + 1) + " finished!")
    return all_losses

# Reused from task 1 for consistency 
# This function is used to evaluate the model
def evaluate_model(regression_model, test_dataloader):
    regression_model.eval()
    all_preds = []
    all_labels = []

    #first we collect all labels and predictions of the test-set for different performance metrics
    for input, attention_mask, labels in test_dataloader:
        input = input.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
            
        #compute predictions for our test-dataset
        outputs = regression_model(input=input, attention_masks=attention_mask)
        
        #store all predictions and labels
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    #compute the MSE, the MAE and the R2-score
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    return mse, mae, r2, all_preds, all_labels

# This function combines original training dataset and newly selected K samples
def merge_tokenized_datasets(selected_samples, pre_train_dataset):
    print(f"Original Lipophilicity Training Dataset Size: {len(pre_train_dataset)}")
    print(f"Number of Selected Samples: {len(selected_samples)}")

    # Define empty lists
    selected_inputs = []
    selected_masks = []
    selected_labels = []
    
    # Iterate through the selected dataset to collect all items
    for i in range(len(selected_samples)):
        input_ids, attention_mask, label = selected_samples[i]
        selected_inputs.append(input_ids)
        selected_masks.append(attention_mask)
        selected_labels.append(label)
    
    # Convert lists to tensors
    selected_inputs = torch.stack(selected_inputs)
    selected_masks = torch.stack(selected_masks)
    selected_labels = torch.stack(selected_labels)
    selected_dataset = TensorDataset(selected_inputs, selected_masks, selected_labels)

    # combine with Lipophilicity training dataset
    combined_train_dataset = ConcatDataset([pre_train_dataset, selected_dataset])
    print(f"Final Merged Training Dataset Size: {len(combined_train_dataset)}")

    return combined_train_dataset

# Reused from task 1 for consistency 
# This function computes the margin
def compute_margin(labels, predictions):
    errors = np.abs(predictions - labels)
    margin = np.median(errors)
    return margin

# main function starts here
if __name__ == "__main__":
    # define rge device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
        
    # Load the model and tokenizer and put it in evaluation mode for influence score calculation
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    
    # Load main dataset and split it into training and testing dataset
    dataset = load_dataset(DATASET_PATH)
    compl_dataset = dataset['train']
    split_dataset = compl_dataset.train_test_split(test_size=0.2)
    pre_train_dataset = split_dataset['train']
    pre_test_dataset = split_dataset['test']
    
    # Load external dataset
    ext_data = pd.read_csv("/home/neuronet_team130/nnti_project/External-Dataset_for_Task2.csv")

    # Convertall 3 datasets to necessar format using the defined class SMILESDataset
    train_dataset = SMILESDataset(pre_train_dataset, tokenizer)
    test_dataset = SMILESDataset(pre_test_dataset, tokenizer)
    external_dataset = SMILESDataset(ext_data, tokenizer)

    # define the loss function
    loss_fn = torch.nn.MSELoss()
    
    # calculate the influence scores
    influence_scores = compute_influence(
        model, test_dataset, external_dataset, loss_fn,
        recursion_depth=RECURSION_DEPTH, damping=DAMPING, batch_size=BATCH_SIZE
    )

    # Save the influence scores for intermediate evaluation and also get top k samples for further process
    results_df, selected_samples = save_influence_results(influence_scores, external_dataset, "lipophilicity_influence_scores.csv")

    # print Top influence data points
    print("Top influence data points are: ")
    print(results_df.head())

    # Convert the selected top k samples to necessary format and load the data using DataLoader
    selected_dataset = SMILESDataset(selected_samples, tokenizer)
    Sample_dataloader = DataLoader(selected_dataset, batch_size=20, shuffle=True)
 
    # merge this samples with training samples and load the merged dataset
    merged_train_dataset = merge_tokenized_datasets(selected_dataset, train_dataset) 
    merged_train_dataloader = DataLoader(merged_train_dataset, batch_size=20, shuffle=True)

    # Also load the test and train dataset
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    # load the pretrained model for training with merged dataset
    mlm_model = AutoModel.from_pretrained( MODEL_PATH, deterministic_eval=True, trust_remote_code=True).to(device)
    mlm_regression_model = MoLFormerWithRegressionHead(mlm_model).to(device)

    # train the model and evaluate it
    mlm_losses = train_model(regression_model=mlm_regression_model, train_dataloader=merged_train_dataloader, epochs=20)

    mlm_mse, mlm_mae, mlm_r2, mlm_all_preds, mlm_all_labels = evaluate_model(regression_model=mlm_regression_model, test_dataloader=test_dataloader)

    mlm_x_min, mlm_x_max = min(mlm_all_labels), max(mlm_all_preds)

    # compute margins for median of the points
    mlm_margin = compute_margin(mlm_all_labels, mlm_all_preds)

    # Load the pretraied model again to train with original training datapoints for comparision of metrics - training necessary as regression head is added now
    mlm_regression_model_original = MoLFormerWithRegressionHead(mlm_model).to(device)

    # train the model and evaluate it
    mlm_losses_original = train_model(regression_model=mlm_regression_model_original, train_dataloader=train_dataloader, epochs=20)

    mlm_mse_original, mlm_mae_original, mlm_r2_original, mlm_all_preds_original, mlm_all_labels_original = evaluate_model(regression_model=mlm_regression_model_original, test_dataloader=test_dataloader)

    x_min_original, x_max_original = min(mlm_all_labels_original), max(mlm_all_preds_original)

    # compute margins for median of the points
    margin_original = compute_margin(mlm_all_labels_original, mlm_all_preds_original)

    metrics = pd.DataFrame({
        "Metric": ["MSE", "MAE", "R² Score", "Margin for Median"],
        "Baseline Model": [mlm_mse, mlm_mae, mlm_r2, mlm_margin],
        "Influence based data selected model": [mlm_mse_original, mlm_mae_original, mlm_r2_original, margin_original]
    })

    print(metrics)

    # reused from task 1 for consistency with necessary changes

    #plot each model next to each other
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(mlm_all_labels_original, mlm_all_preds_original, alpha=0.5, color="blue")
    #ideal prediction line with margin of median
    axes[0].plot(mlm_all_labels_original, mlm_all_labels_original, 'r--', label="Ideal Prediction Line")
    axes[0].fill_between(
        [x_min_original, x_max_original], 
        [x_min_original - margin_original * x_max_original, x_max_original - margin_original * x_max_original], 
        [x_min_original + margin_original * x_max_original, x_max_original + margin_original * x_max_original], 
        color='red', alpha=0.1, label="Median Margin"
    )
    axes[0].set_title("Pretrained Model")
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend()

    axes[1].scatter(mlm_all_labels, mlm_all_preds, alpha=0.5, color="orange")
    #ideal prediction line
    axes[1].plot(mlm_all_labels, mlm_all_labels, 'r--', label="Ideal Prediction Line")
    axes[1].fill_between(
        [mlm_x_min, mlm_x_max], 
        [mlm_x_min - mlm_margin * mlm_x_max, mlm_x_max - mlm_margin * mlm_x_max], 
        [mlm_x_min + mlm_margin * mlm_x_max, mlm_x_max + mlm_margin * mlm_x_max], 
        color='red', alpha=0.1, label="Median Margin"
    )
    axes[1].set_title("Influence Function-based Data Selection Model")
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    plt.show()
    plt.close(fig)

    #plot the average losses for each epoch of training for each model
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(mlm_losses_original) + 1), mlm_losses_original, marker='o', linestyle='-', color="blue", label="Pretrained Model")
    ax.plot(range(1, len(mlm_losses) + 1), mlm_losses, marker='o', linestyle='-', color="orange", label="Influence Function-based Data Selection Model")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Training Loss Over Epochs")
    ax.grid(True)
    ax.legend()
    plt.savefig("loss_plot.png")
    plt.show()
    plt.close(fig2)
