# import dependencies
import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"
DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
ext_data = pd.read_csv("/home/neuronet_team100/nnti_project/tasks/External-Dataset_for_Task2.csv")


########################################################
# Entry point
########################################################
# define a PyTorch Dataset class for handling SMILES strings and targets

def evaluate_model(regression_model, test_dataloader):
    regression_model.eval()
    all_preds = []
    all_labels = []

    # first we collect all labels and predictions of the test-set for different performance metrics
    for input, attention_mask, labels in test_dataloader:
        input = input.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # compute predictions for our test-dataset
        outputs = regression_model(input=input, attention_masks=attention_mask)

        # store all predictions and labels
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # compute the MSE, the MAE and the R2-score
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return mse, mae, r2, all_preds, all_labels


def compute_margin(labels, predictions):
    errors = np.abs(predictions - labels)
    margin = np.median(errors)
    return margin


class ExternalDataclass(Dataset):
    """
        Custom Dataclass
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # the SMILE stings first have to be tokenized for the model
        row = self.dataset.iloc[index]

        # Convert label to tensor
        label = torch.tensor(row["Label"], dtype=torch.float)

        # the SMILE stings first have to be tokenized for the model
        smiles = row["SMILES"]
        inputs_with_mask = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
            return_attention_mask=True
        )

        input_1 = inputs_with_mask["input_ids"].squeeze(0)
        attention_mask = inputs_with_mask["attention_mask"].squeeze(0)

        return input_1, attention_mask, label

    def __len__(self):
        return len(self.dataset)


class ExternalDataLoader(DataLoader):
    """
        Custom Dataloader
    """

    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        for batch in super().__iter__():
            yield batch


def mc_dropout_predictions(model, dataloader, num_mc_passes=50, device='cuda'):
    """

    :param model: relevant model
    :param dataloader: external dataset
    :param num_mc_passes: number of forward passes to make through the model to get variance
    :param device: device
    :return: list of predicted values for each data sample from each forward pass
    """
    model.train()  # Keep dropout enabled during inference
    all_predictions = []

    with torch.no_grad():
        for input, attention_mask, _ in dataloader:
            input, attention_mask = input.to(device), attention_mask.to(device)

            preds = []
            for _ in range(num_mc_passes):
                output = model(input, attention_mask)
                preds.append(output.cpu().numpy())

            all_predictions.append(np.stack(preds))

    return np.array(all_predictions)


def compute_uncertainty(mc_predictions):
    """

    :param mc_predictions: output from mc_dropout_predictions function
    :return: variance
    """
    return np.var(mc_predictions, axis=1)


def filter_high_uncertainty_data(uncertainty_scores, dataset, percentile=90):
    """
    Filters data points with uncertainty above the given percentile and returns a dictionary
    containing input_ids, attention_mask, and labels.

    Args:
        uncertainty_scores (np.array): Array of uncertainty values for each data point.
        dataset (torch.utils.data.Dataset): Dataset containing pre-tokenized samples.
        percentile (int, optional): Percentile threshold for high uncertainty. Defaults to 90.

    Returns:
        dict: Dictionary with keys 'input_ids', 'attention_mask', and 'labels'.
    """
    threshold = np.percentile(uncertainty_scores, percentile)
    # threshold = np.mean(uncertainty_scores)
    high_uncertainty_indices = np.where(uncertainty_scores > threshold)[0]
    selected_samples = {'input_ids': [], 'attention_mask': [], 'labels': []}

    # Extract filtered data
    for idx in high_uncertainty_indices:
        sample = dataset[idx]
        selected_samples['input_ids'].append(sample[0])  # input_ids
        selected_samples['attention_mask'].append(sample[1])  # attention_mask
        selected_samples['labels'].append(sample[2])

    print("number of selected samples: ", len(selected_samples['input_ids']))

    return selected_samples


class SMILESDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # put the labels in a tensor for the model
        label = torch.tensor(self.dataset[idx]["label"], dtype=torch.float)

        # the SMILE stings first have to be tokenized for the model
        smiles = self.dataset[idx]["SMILES"]
        inputs_with_mask = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
            return_attention_mask=True
        )

        input = inputs_with_mask["input_ids"].squeeze(0)
        attention_mask = inputs_with_mask["attention_mask"].squeeze(0)

        return input, attention_mask, label


class LoRA(nn.Module):
    """
    LoRA class
    Accepts a linear layer and adjusts with LoRA layer
    """
    def __init__(self, rank, alpha, out_features, in_features, linear, device='cpu'):
        super().__init__()
        standard_deviation = 1 / torch.sqrt(torch.tensor(rank).float())
        self.B = nn.Parameter(torch.randn(in_features, rank, device=device) * standard_deviation)
        self.A = nn.Parameter(torch.zeros(rank, out_features, device=device))
        self.rank = rank
        self.alpha = alpha

        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        # returns X(W + W'*scale)
        delta_w = torch.matmul(self.B, self.A)
        w = self.linear.weight + (self.alpha / self.rank) * delta_w
        return torch.nn.functional.linear(x, w, self.linear.bias)


class MoLFormerWithRegressionHead(nn.Module):
    def __init__(self, curr_model, rank, alpha, bitfit=False, use_lora=False):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.model = curr_model
        self.dropout = nn.Dropout(p=0.1)  # 10% dropout
        # a single linear layer we use for the regression head
        # 768 is common for transformers
        self.regression_head = nn.Linear(768, 1)
        self.enable_dropout = False
        self.bitfit = bitfit
        self.use_lora = use_lora

        if self.bitfit:
            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # unfreeze only bias parameters
            for module in self.model.modules():
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.requires_grad = True

            for param in self.regression_head.parameters():
                param.requires_grad = True

            # Number of trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

        if self.use_lora:
            original_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            for layer in self.model.encoder.layer:
                layer.attention.self.query = LoRA(rank, alpha, 768, 768, layer.attention.self.query)
                layer.attention.self.value = LoRA(rank, alpha, 768, 768, layer.attention.self.value)

            # unfreeze LoRA matrices B and A
            for param in layer.attention.self.query.parameters():
                if param is layer.attention.self.query.A or param is layer.attention.self.query.B:
                    param.requires_grad = True
            for param in layer.attention.self.value.parameters():
                if param is layer.attention.self.value.A or param is layer.attention.self.value.B:
                    param.requires_grad = True

            for param in self.regression_head.parameters():
                param.requires_grad = True

            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            regression_head_params = sum(p.numel() for p in self.regression_head.parameters() if p.requires_grad)
            total_trainable_params = lora_params + regression_head_params
            print(f"Original trainable parameters: {original_params:,}")
            print(f"Trainable parameters after LoRA (including regression head): {total_trainable_params:,}")
            print(f"Reduction in trainable parameters: {original_params - total_trainable_params:,}")

    def forward(self, input, attention_masks):
        outputs_molformer = self.model(input_ids=input, attention_mask=attention_masks)
        pooled_output = outputs_molformer.last_hidden_state[:, 0, :]

        if self.enable_dropout:  # dropout flag
            pooled_output = self.dropout(pooled_output)

        regression_output = self.regression_head(pooled_output)

        return regression_output.squeeze(-1)

    def activate_mc_dropout(self):
        """Enable dropout during inference for MC Dropout"""
        self.enable_dropout = True

    def deactivate_mc_dropout(self):
        """Disable dropout during normal inference and fine-tuning"""
        self.enable_dropout = False


class iA3Transformer(nn.Module):
    """
    iA3 class
    Same as the MolFormer class, adjusted for the scaling tensors with a new forward function.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

        num_layers = len(self.model.encoder.layer)
        hidden_dim = self.model.config.hidden_size
        self.regression_head = nn.Linear(hidden_dim, 1)
        original_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # scaling parameters
        self.l_k = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim) * 0.8) for _ in range(num_layers)])
        self.l_v = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim) * 0.8) for _ in range(num_layers)])

        # freeze all model parameters except iA3 parameters
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.l_k:
            param.requires_grad = True

        for param in self.l_v:
            param.requires_grad = True

        for param in self.regression_head.parameters():
            param.requires_grad = True

        iA3_params_lv = sum(p.numel() for p in self.l_v if p.requires_grad)
        iA3_params_lk = sum(p.numel() for p in self.l_k if p.requires_grad)
        regression_head_params = sum(p.numel() for p in self.regression_head.parameters() if p.requires_grad)
        total_trainable_params = iA3_params_lk + iA3_params_lv + regression_head_params
        print(f"Original trainable parameters: {original_params:,}")
        print(f"Trainable parameters after iA3 (including regression head): {total_trainable_params:,}")
        print(f"Reduction in trainable parameters: {original_params - total_trainable_params:,}")

    def forward(self, input, attention_masks):
        outputs = self.model(input_ids=input, attention_mask=attention_masks, output_hidden_states=True)

        hidden_states = outputs.last_hidden_state

        for i, layer in enumerate(self.model.encoder.layer):
            attention_layer = layer.attention.self

            # iA3 implementation: softmax(Q(l_k * K)/(sq root d))(l_v * v)

            Q = attention_layer.query(hidden_states)
            K = attention_layer.key(hidden_states)
            V = attention_layer.value(hidden_states)

            K = K * self.l_k[i].view(1, 1, -1)
            V = V * self.l_v[i].view(1, 1, -1)

            d_k = K.size(-1)
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

            hidden_states = torch.matmul(attn_weights, V)

        pooled_output = hidden_states[:, 0, :]
        regression_output = self.regression_head(pooled_output)
        return regression_output.squeeze(-1)


# we define a function for training the model with regression head
def train_model(regression_model, train_dataloader, lr=2e-5, epochs=5, use_bitfit=False, use_lora=False, use_iA3=False):
    # standard optimizer and loss for transformers and regression
    """
    Same as the train_model code from task 1. Adjusted for BitFit/LoRA/iA3 tasks.
    """
    if use_bitfit:
        # only optimize the BitFit parameters
        optimizer = torch.optim.AdamW(
            [p for p in regression_model.parameters() if p.requires_grad], lr=1e-4
        )
    elif use_lora:
        # only optimize the LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in regression_model.parameters() if p.requires_grad], lr=2e-5
        )
    elif use_iA3:
        # only optimize the iA3 parameters
        optimizer = torch.optim.AdamW(
            [p for p in regression_model.parameters() if p.requires_grad], lr=1e-3
        )
    else:
        # standard optimization
        optimizer = torch.optim.AdamW(regression_model.parameters(), lr=lr)

    """
    This is the end of relevant new code added
    """

    criterion = nn.MSELoss()
    regression_model.train()
    all_losses = []

    for epoch in range(epochs):
        # additionally compute avg loss for each epoch for evaluation
        total_loss = 0
        count_batch = 0

        # for each epoch run trough all training samples
        for input, attention_mask, labels in train_dataloader:
            # for each training sample run forward and back propagate trough our model
            input = input.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # run forward
            outputs = regression_model(input=input, attention_masks=attention_mask)
            # compute loss and backpropagate
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # sum up losses
            total_loss += loss.item()
            count_batch += 1

        # compute and store avg loss for every epoch
        avg_loss = total_loss / count_batch
        all_losses.append(avg_loss)

        # print("Training-Epoch Nr." + str(epoch + 1) + " finished!")
    return all_losses


def metric_output(regression_model, test_dataloader, losses, base_pred_name, average_loss_name):
    """
    Model evaluation code - SAME AS TASK 1

    :param regression_model: model that we are evaluating
    :param test_dataloader: the test dataloader
    :param losses: train_model function output
    :param base_pred_name: destination file for true vs predicted graph
    :param average_loss_name: destination file for average training loss over epochs graph
    :return: N.A.
    """
    mse, mae, r2, all_preds, all_labels = evaluate_model(regression_model=regression_model,
                                                         test_dataloader=test_dataloader)
    x_min, x_max = min(all_labels), max(all_preds)
    # compute margins for median of the points
    margin = compute_margin(all_labels, all_preds)
    print(f"Mean Squared Error (MSE):", mse)
    print(f"Mean Absolute Error (MAE):", mae)
    print(f"R² Score:", r2)
    print(f"Margin for Median:", margin)
    # we plot predictions and labels to see how our regression model performs
    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    # additional ideal prediction line
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r')
    plt.fill_between(
        [x_min, x_max],
        [x_min - margin * x_max, x_max - margin * x_max],
        [x_min + margin * x_max, x_max + margin * x_max],
        color='red', alpha=0.1, label="Margin"
    )
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted")
    plt.savefig(base_pred_name)
    plt.show()
    # plot the average losses for each epoch of training
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.savefig(average_loss_name)
    plt.show()


def merge_tokenized_datasets(selected_samples, pre_train_dataset):
    """
    Merges the selected tokenized samples with the existing Lipophilicity training dataset.
    Returns a new dataset that can be directly used in a DataLoader.
    """
    print(f"Original Lipophilicity Training Dataset Size: {len(pre_train_dataset)}")
    print(f"Number of Selected Samples: {len(selected_samples['input_ids'])}")

    # convert into tensors
    selected_inputs = torch.stack(selected_samples['input_ids'])
    selected_masks = torch.stack(selected_samples['attention_mask'])
    selected_labels = torch.stack(selected_samples['labels'])
    selected_dataset = TensorDataset(selected_inputs, selected_masks, selected_labels)

    # combine with Lipophilicity training dataset
    combined_train_dataset = ConcatDataset([pre_train_dataset, selected_dataset])
    print(f"Final Merged Training Dataset Size: {len(combined_train_dataset)}")

    return combined_train_dataset


def merge_tokenized_datasets_mlm(selected_samples, pre_train_dataset):
    """
    Mlm version for merge tokenized datasets
    """
    print(f"Original Lipophilicity Training Dataset Size: {len(pre_train_dataset)}")
    print(f"Number of Selected Samples: {len(selected_samples['input_ids'])}")

    # convert into tensors
    selected_inputs = torch.stack(selected_samples['input_ids'])
    selected_masks = torch.stack(selected_samples['attention_mask'])
    selected_dataset = TensorDataset(selected_inputs, selected_masks)

    # combine with Lipophilicity training dataset
    combined_train_dataset = ConcatDataset([pre_train_dataset, selected_dataset])
    print(f"Final Merged MLM Training Dataset Size: {len(combined_train_dataset)}")

    return combined_train_dataset


class DictWrapperDataset(Dataset):
    """
    Wraps ConcatDataset from filter_high_uncertainty_data_mlm and converts samples to dictionaries for
    MLM fine-tuning (no labels). Used for providing proper input to the mlm dataloader.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # convert tuple (TensorDataset output) to dictionary
        if isinstance(sample, tuple):
            return {"input_ids": sample[0], "attention_mask": sample[1]}  # No labels

        return sample


def tokenize_function(examples):
    return tokenizer(examples["SMILES"], truncation=True, padding="max_length", max_length=128)


if __name__ == "__main__":
    """
    Part 1: Get the pretrained model and train it on the training dataset of Lipophilicity. This is code from task 1, 
    nothing new has been added for this part.
    """
    # load the dataset from HuggingFace
    dataset = load_dataset(DATASET_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # split the data into training and test datasets
    # we split the dataset with a ratio of 0.2 for testing and 0.8 for training
    compl_dataset = dataset['train']
    split_dataset = compl_dataset.train_test_split(test_size=0.2)

    pre_train_dataset = split_dataset['train']
    pre_test_dataset = split_dataset['test']

    # we also tokenize the data with our SMILESDataset-class
    train_dataset = SMILESDataset(pre_train_dataset, tokenizer)
    test_dataset = SMILESDataset(pre_test_dataset, tokenizer)

    BATCH_SIZE = 20
    base_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # load pre-trained model from HuggingFace
    base_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

    # initialize the regression model
    base_regression_model = MoLFormerWithRegressionHead(base_model, 0, 0).to(device)

    # train our model with regression head
    base_regression_model.deactivate_mc_dropout()
    base_losses = train_model(regression_model=base_regression_model, train_dataloader=base_train_dataloader, epochs=20)

    """
    Part 2.1: External dataset prep + tokenization
    """
    external_dataset = ExternalDataclass(ext_data, tokenizer)
    external_train_loader = ExternalDataLoader(external_dataset, batch_size=1, shuffle=True)
    base_regression_model.activate_mc_dropout()

    """
    Part 2.2: Uncertainty Estimation with MC Dropout; we get the selected samples based on threshold
    """
    mc_preds = mc_dropout_predictions(base_regression_model, external_train_loader)
    uncertainty_scores = compute_uncertainty(mc_preds)
    base_regression_model.deactivate_mc_dropout()

    sorted_indices = np.argsort(uncertainty_scores, axis=0)
    sorted_uncertainty = np.take_along_axis(uncertainty_scores, sorted_indices, axis=0).flatten()
    mean_uncertainty = np.mean(uncertainty_scores)

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_uncertainty, alpha=0.7, color='b', label="Uncertainty (Variance)")

    threshold = np.percentile(uncertainty_scores, 90)
    threshold_1 = np.percentile(uncertainty_scores, 60)
    plt.axhline(y=threshold, color='r', linestyle="--", label=f"90th Percentile: {threshold:.4f}")
    plt.axhline(y=threshold_1, color='green', linestyle="--", label=f"60th Percentile: {threshold_1:.4f}")
    plt.axhline(y=mean_uncertainty, color='purple', linestyle="--", label=f"50th Percentile: {mean_uncertainty:.4f}")

    plt.xlabel("Sorted Data Point Index")
    plt.ylabel("Variance (Uncertainty)")
    plt.title("Sorted Uncertainty (Variance) vs. Data Point Index")
    plt.legend()
    plt.grid(True)

    plt.savefig("variance.png")
    plt.show()

    # selected samples
    selected_dataset = filter_high_uncertainty_data(uncertainty_scores, external_dataset, 90)

    """
    Part 3: Combine selected data from external dataset with lipophilicity training dataset and train model again. 
    """

    merged_train_dataset = merge_tokenized_datasets(selected_dataset, train_dataset)  # merge with selected data
    train_dataloader = DataLoader(merged_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)

    regression_model = MoLFormerWithRegressionHead(model, 0, 0).to(device)
    losses = train_model(regression_model=regression_model, train_dataloader=train_dataloader, epochs=20)

    mse, mae, r2, all_preds, all_labels = evaluate_model(regression_model=regression_model,
                                                         test_dataloader=test_dataloader)

    x_min, x_max = min(all_labels), max(all_preds)

    # compute margins for median of the points
    margin = compute_margin(all_labels, all_preds)

    print(f"Mean Squared Error (MSE):", mse)
    print(f"Mean Absolute Error (MAE):", mae)
    print(f"R² Score:", r2)
    print(f"Margin for Median:", margin)
    metric_output(regression_model, test_dataloader, losses, "uncertainty_true_vs_predicted.png",
                  "uncertainty_avr_train_loss.png")

    """
    Part 4: Fine-tuning on new dataset. Only relevant code here is feeding the merged tokenized dataset to the mlm
    fine-tuning model. The rest of the code is same as code from task 1.
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    tokenized_dataset = compl_dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["SMILES", "label"])

    # again split our tokenized data into train and test data
    mlm_split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    mlm_train_dataset = mlm_split_dataset['train']
    mlm_test_dataset = mlm_split_dataset['test']

    # construct DataLoaders with our defined masking for both train and test datasets
    merged_mlm_train_dataset = DictWrapperDataset(
        merge_tokenized_datasets_mlm(selected_dataset, mlm_train_dataset))  # merge with selected data
    train_dataloader_masked = DataLoader(merged_mlm_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=data_collator)
    """
    End of relevant code for mlm fine-tuning
    """
    test_dataloader_masked = DataLoader(mlm_test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

    # set up the original model with the MLM-objective
    mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    mlm_model.to(device)

    # train the mlm-model
    epochs = 20
    lr = 2e-5
    # standard optimizer and loss for transformers and multi-classification
    optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    mlm_model.train()

    for epoch in range(epochs):
        # for each epoch run trough all trainingssamples
        for batch in train_dataloader_masked:
            # for each trainingssample run forward and backpropagate trough our model
            input = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if "labels" in batch:
                labels = batch["labels"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            # run forward
            outputs = mlm_model(input_ids=input, attention_mask=attention_mask)

            # flatten the logits and labels
            logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
            labels_flat = labels.view(-1)

            # compute loss and backpropagate
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

        print("Training-Epoch Nr." + str(epoch + 1) + " finished!")

    # save the trained mlm-model
    mlm_model.save_pretrained("molformer_mlm")

    # again define pretrained mlm_model
    mlm_model = AutoModel.from_pretrained("molformer_mlm", deterministic_eval=True, trust_remote_code=True)

    # put the regression head on the pretrained mlm_model
    mlm_regression_model = MoLFormerWithRegressionHead(mlm_model, 0, 0).to(device)

    # since we can use the dataloader from the first regression-model, we can diectly start training
    mlm_losses = train_model(regression_model=mlm_regression_model, train_dataloader=train_dataloader, epochs=20)

    # evaluating and comparing the only finetuned regression model with the additional mlm-pretrained and finetuned regression model
    mlm_mse, mlm_mae, mlm_r2, mlm_all_preds, mlm_all_labels = evaluate_model(regression_model=mlm_regression_model,
                                                                             test_dataloader=test_dataloader)
    mse, mae, r2, all_preds, all_labels = evaluate_model(regression_model=regression_model,
                                                         test_dataloader=test_dataloader)

    # compute margins for median of the points
    margin = compute_margin(all_labels, all_preds)
    mlm_margin = compute_margin(mlm_all_labels, mlm_all_preds)

    # compute min of labels and max of predictions for plotting the margins
    x_min, x_max = min(all_labels), max(all_preds)
    mlm_x_min, mlm_x_max = min(mlm_all_labels), max(mlm_all_preds)

    # plot each model next to each other
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(all_labels, all_preds, alpha=0.5, color="blue")
    # ideal prediction line with margin of median
    axes[0].plot(all_labels, all_labels, 'r--', label="Ideal Prediction Line")
    axes[0].fill_between(
        [x_min, x_max],
        [x_min - margin * x_max, x_max - margin * x_max],
        [x_min + margin * x_max, x_max + margin * x_max],
        color='red', alpha=0.1, label="Median Margin"
    )
    axes[0].set_title("Only Regression Model")
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].legend()

    axes[1].scatter(mlm_all_labels, mlm_all_preds, alpha=0.5, color="orange")
    # ideal prediction line
    axes[1].plot(mlm_all_labels, mlm_all_labels, 'r--', label="Ideal Prediction Line")
    axes[1].fill_between(
        [mlm_x_min, mlm_x_max],
        [mlm_x_min - mlm_margin * mlm_x_max, mlm_x_max - mlm_margin * mlm_x_max],
        [mlm_x_min + mlm_margin * mlm_x_max, mlm_x_max + mlm_margin * mlm_x_max],
        color='red', alpha=0.1, label="Median Margin"
    )
    axes[1].set_title("MLM-Pretrained Model")
    axes[1].set_xlabel("True Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].legend()

    # create table for comparison of the metrics
    metrics = pd.DataFrame({
        "Metric": ["MSE", "MAE", "R² Score", "Margin for Median"],
        "Original Model": [mse, mae, r2, margin],
        "MLM-pretrained Model": [mlm_mse, mlm_mae, mlm_r2, mlm_margin]
    })

    print(metrics)
    plt.tight_layout()
    plt.savefig("metrics_task3a.png")
    plt.show()

    # plot the average losses for each epoch of training for each model
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color="blue", label="Only Regression Model")
    plt.plot(range(1, len(losses) + 1), mlm_losses, marker='o', linestyle='-', color="orange",
             label="MLM-Pretrained Model")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig("fine_tuned_task_3a.png")
    plt.show()

    """
    Part 5: BitFit, LoRA and iA3 implementation on the new dataset we made using uncertainty based data sampling
    """
    print("------------------------------------------------------------------------------------------------------")
    print("BitFit Results")
    bitfit_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    bitfit_regression_model = MoLFormerWithRegressionHead(bitfit_model, 0, 0, bitfit=True).to(device)
    bitfit_losses = train_model(regression_model=bitfit_regression_model, train_dataloader=train_dataloader,
                                epochs=40,
                                use_bitfit=True)
    metric_output(bitfit_regression_model, test_dataloader, bitfit_losses, "bitfit_true_vs_predicted.png",
                  "bitfit_average_training_loss.png")

    print("------------------------------------------------------------------------------------------------------")
    print("LoRA Results")
    lora_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    lora_regression_model = MoLFormerWithRegressionHead(base_model, 4, 64, bitfit=False, use_lora=True).to(device)
    lora_model_losses = train_model(lora_regression_model, train_dataloader, epochs=40, use_lora=True)
    metric_output(lora_regression_model, test_dataloader, lora_model_losses, "lora_true_vs_predicted.png",
                  "lora_average_training_loss.png")

    print("------------------------------------------------------------------------------------------------------")
    print("iA3 Results")
    iA3_model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    iA3_regression_model = iA3Transformer(iA3_model).to(device)
    iA3_model_losses = train_model(iA3_regression_model, train_dataloader, epochs=20, use_iA3=True)
    metric_output(iA3_regression_model, test_dataloader, iA3_model_losses, "iA3_true_vs_predicted.png",
                  "iA3_average_training_loss.png")
