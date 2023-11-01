## Part 1: Building Vision Transformers from Scratch: A PyTorch Deep Dive Plus a Teaser on LORA for Part 2

If you've delved into the realm of deep learning, you're likely aware of the  impact that transformer architectures have had on the field of artificial intelligence. These architectures stand at the core of numerous groundbreaking advancements in AI. In this Article, we will embark on an in-depth exploration, guiding you through the process of building Vision Transformers from the ground up.

This article is the first in a four-part series. The next one will show how to build 'LoRa' from scratch, for the Vision Transformer we are building here.


```python
import math
import torch 
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch import nn
from dataclasses import dataclass
import torch.nn.functional as F
```


```python
# Define a configuration for the model using a data class
@dataclass
class ModelArgs:
    dim: int = 256          # Dimension of the model embeddings
    hidden_dim: int = 512   # Dimension of the hidden layers
    n_heads: int = 8        # Number of attention heads
    n_layers: int = 6       # Number of layers in the transformer
    patch_size: int = 4     # Size of the patches (typically square)
    n_channels: int = 3     # Number of input channels (e.g., 3 for RGB images)
    n_patches: int = 64     # Number of patches in the input
    n_classes: int = 10     # Number of target classes
    dropout: float = 0.2    # Dropout rate for regularization

```

## **MultiHead Attention Overview**

The `MultiHeadAttention` module in the provided code is an implementation of the multi-head self-attention mechanism, which stands as a fundamental component in transformer architectures. This self-attention mechanism empowers the model to weigh input elements differently, offering the capability to focus more intently on certain parts of the input when generating the output.

First, let's code the multi-head attention block. Afterward, I'll break down the key components in detail


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # Linear projections for Q, K, and V
        self.wq = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads*self.head_dim, self.dim, bias=False)
    
    def forward(self, x):
        b, seq_len, dim = x.shape  # b: batch size, seq_len: sequence length
        
        assert dim == self.dim, "dim is not matching"
        
        q = self.wq(x)  # [b, seq_len, n_heads*head_dim]
        k = self.wk(x)  # [b, seq_len, n_heads*head_dim]
        v = self.wv(x)  # [b, seq_len, n_heads*head_dim]
        
        # Reshape the tensors for multi-head operations
        q = q.contiguous().view(b, seq_len, self.n_heads, self.head_dim)  # [b, seq_len, n_heads, head_dim]
        k = k.contiguous().view(b, seq_len, self.n_heads, self.head_dim)  # [b, seq_len, n_heads, head_dim]
        v = v.contiguous().view(b, seq_len, self.n_heads, self.head_dim)  # [b, seq_len, n_heads, head_dim]
        
        # Transpose to bring the head dimension to the front
        q = q.transpose(1, 2)  # [b, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [b, n_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [b, n_heads, seq_len, head_dim]
        
        # Compute attention scores and apply softmax
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # [b, n_heads, seq_len, seq_len]
        attn_scores = F.softmax(attn, dim=-1)  # [b, n_heads, seq_len, seq_len]
        
        # Compute the attended features
        out = torch.matmul(attn_scores, v)  # [b, n_heads, seq_len, head_dim]
        out = out.contiguous().view(b, seq_len, -1)  # [b, seq_len, n_heads*head_dim]
        
        return self.wo(out)  # [b, seq_len, dim]

```

The MultiHeadAttention module performing the following operations:
1. Linear transformations of the input tensor into **"query" (Q)**, **"key" (K)**, and **"value" (V)** representations.
    ```python
       q = self.wq(x)
       k = self.wk(x)
       v = self.wv(x)
    ```
2. Dividing these tensors into multiple "heads".
    ```python
        q = q.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
        k = k.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
        v = v.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
    ```
3. Computing attention scores via the dot product of Q and K.
    ```python
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
    ```
4. Applying softmax to these scores to procure attention weights.
    ```python
        attn_scores = F.softmax(attn, dim=-1)
    ```
5. Multiplying the attention weights with the V tensor, yielding the attended features.
    ```python
        out = torch.matmul(attn_scores, v)
    ```
6. Aggregating results across all heads and projecting to provide the concluding output.
    ```python
        out = out.contiguous().view(b, seq_len, -1)
        return self.wo(out)
    ```

## **AttentionBlock Overview**

The `AttentionBlock` module encapsulates a typical block found within transformer architectures. It primarily consists of two significant components: a multi-head self-attention mechanism and a feed-forward neural network (FFN). Additionally, layer normalization and skip connections (residual connections) are employed to facilitate better learning and gradient flow.

let's code the multi-head attention block.




```python
class AttentionBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(args.dim)
        self.attn = MultiHeadAttention(args)
        
        self.layer_norm_2 = nn.LayerNorm(args.dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(args.dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.dim),
            nn.Dropout(args.dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.layer_norm_1(x))
        x = x + self.ffn(self.layer_norm_2(x))
        return x
```

Let's delve deeper into its structure:

1. **Layer Normalization (Pre-Attention)**
   Before feeding the input `x` into the multi-head attention mechanism, it's normalized using `LayerNorm`.
   ```python
   self.layer_norm_1 = nn.LayerNorm(args.dim)
   x = self.layer_norm_1(x)
   ```

2. **Multi-Head Self-Attention**
   This component allows the model to focus on different parts of the input sequence when generating its output.
   ```python
   self.attn = MultiHeadAttention(args)
   x = x + self.attn(x)
   ```

3. **Layer Normalization (Pre-Feed-Forward Network)**
   Just like before the multi-head attention mechanism, the output is normalized again using `LayerNorm` before feeding it into the FFN.
   ```python
   self.layer_norm_2 = nn.LayerNorm(args.dim)
   x = self.layer_norm_2(x)
   ```

4. **Feed-Forward Neural Network (FFN)**
   The FFN consists of two linear layers separated by a GELU activation function. There's also dropout applied for regularization.
   ```python
   self.ffn = nn.Sequential(
       nn.Linear(args.dim, args.hidden_dim),
       nn.GELU(),
       nn.Dropout(args.dropout),
       nn.Linear(args.hidden_dim, args.dim),
       nn.Dropout(args.dropout)
   )
   x = x + self.ffn(x)
   ```

5. **Residual Connections**
   Residual or skip connections are vital for deep architectures like transformers. They help in preventing the vanishing gradient problem and aid in model convergence. In the code, these are represented by the addition operations where the input is added back to the output of both the attention mechanism and the FFN.
   ```python
   x = x + self.attn(...)
   x = x + self.ffn(...)
   ```

By sequentially organizing the operations, this block ensures efficient and effective feature transformation, which is essential for the transformer's performance.


Before creating our full vistion transformer model, we need to create a utility function that transforms images into non-overlapping patches.


```python
def img_to_patch(x, patch_size, flatten_channels=True):
    # x: Input image tensor 
    # B: Batch size, C: Channels, H: Height, W: Width
    B, C, H, W = x.shape  # (B, C, H, W)
    
    # Reshape the image tensor to get non-overlapping patches
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)  # (B, C, H/patch_size, patch_size, W/patch_size, patch_size)
    
    # Permute to group the patches and channels
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H/patch_size, W/patch_size, C, patch_size, patch_size)
    
    # Flatten the height and width dimensions for patches
    x = x.flatten(1,2)  # (B, (H/patch_size * W/patch_size), C, patch_size, patch_size)
    
    # Option to flatten the channel and spatial dimensions
    if flatten_channels:
        x = x.flatten(2,4)  # (B, (H/patch_size * W/patch_size), (C * patch_size * patch_size))
    
    return x
```

The img_to_patch function takes an image tensor and converts it into non-overlapping patches of a specified size. This operation is typically used in vision transformers to represent an image as a sequence of flattened patches. The function provides an option to flatten the channels or keep them separate.

## **VisionTransformer Overview**

The `VisionTransformer` effectively integrates the previously discussed components to construct the final model. It operates as an encoder-only architecture similar to BERT, where all tokens attend to all other tokens. Moreover, we introduce an additional class token (`cls_token`) to every sequence in the batch, and this will be utilized later for classification, much like how BERT does with its special [CLS] token.



```python
class VisionTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Define the patch size
        self.patch_size = args.patch_size
        
        # Embedding layer to transform flattened patches to desired dimension
        self.input_layer = nn.Linear(args.n_channels * (args.patch_size ** 2), args.dim)

        # Create the attention blocks for the transformer
        attn_blocks = []
        for _ in range(args.n_layers):
            attn_blocks.append(AttentionBlock(args))
        
        # Create the transformer by stacking the attention blocks
        self.transformer = nn.Sequential(*attn_blocks)
        
        # Define the classifier
        self.mlp = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.n_classes)
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(args.dropout)
        
        # Define the class token (similar to BERT's [CLS] token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.dim))
        
        # Positional embeddings to give positional information to transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 1+args.n_patches, args.dim))
    
    def forward(self, x):
        # Convert image to patches and flatten
        x = img_to_patch(x, self.patch_size)
        b, seq_len, _ = x.shape
        
        # Transform patches using the embedding layer
        x = self.input_layer(x)
        
        # Add the class token to the beginning of each sequence
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings to the sequence
        x = x + self.pos_embedding[:,:seq_len+1]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Process sequence through the transformer
        x = self.transformer(x)

        # Retrieve the class token's representation (for classification)
        x = x.transpose(0, 1)
        cls = x[0]

        # Classify using the representation of the class token
        out = self.mlp(cls)
        return out
```

Let's dive into its key aspects:
 
1. **Patch Embedding**
    Instead of operating on raw pixels, the image is divided into fixed-size patches. Each patch is then linearly transformed (flattened and passed through a linear layer) to a specified dimension (`args.dim`).
   ```python
   self.patch_size = args.patch_size
   self.input_layer = nn.Linear(args.n_channels * (args.patch_size ** 2), args.dim)
   x = img_to_patch(x, self.patch_size)  # Assuming `img_to_patch` is a helper function.
   x = self.input_layer(x)
   ```

2. **Transformer Blocks**
   A sequence of attention blocks to process the embedded patches. The number of blocks is defined by `args.n_layers`.
   ```python
   attn_blocks = []
   for _ in range(args.n_layers):
       attn_blocks.append(AttentionBlock(args))
   self.transformer = nn.Sequential(*attn_blocks)
   x = self.transformer(x)
   ```

3. **CLS Token and Position Embeddings**
   A class token is added to the sequence of embedded patches. This token is later used to obtain the final classification output. Positional embeddings are added to provide the transformer with information about the relative positions of patches.
   ```python
   self.cls_token = nn.Parameter(torch.randn(1, 1, args.dim))
   self.pos_embedding = nn.Parameter(torch.randn(1, 1+args.n_patches, args.dim))
   cls_token = self.cls_token.repeat(b, 1, 1)
   x = torch.cat([cls_token, x], dim=1)
   x = x + self.pos_embedding[:,:seq_len+1]
   ```

4. **Dropout**
   Dropout is applied for regularization purposes.
   ```python
   self.dropout = nn.Dropout(args.dropout)
   x = self.dropout(x)
   ```

5. **Classifier**
   The classification head. It uses the class token's[CLS] representation after it's been processed by all transformer blocks.
   ```python
   self.mlp = nn.Sequential(
       nn.LayerNorm(args.dim),
       nn.Linear(args.dim, args.n_classes)
   )
   x = x.transpose(0, 1)
   cls = x[0]
   out = self.mlp(cls)
   ```


The Below code snippet provides a setup for preprocessing and loading the CIFAR10 dataset


```python
# Path to the directory where CIFAR10 data will be stored/downloaded
DATA_DIR = "../data"

# Define the transformation for testing dataset:
# 1. Convert images to tensors.
# 2. Normalize the tensors using the mean and standard deviation of CIFAR10 dataset.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
])

# Define the transformation for training dataset:
# 1. Apply random horizontal flip for data augmentation.
# 2. Perform random resizing and cropping of images for data augmentation.
# 3. Convert images to tensors.
# 4. Normalize the tensors using the mean and standard deviation of CIFAR10 dataset.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
])

# Load the CIFAR10 training dataset with the defined training transformation.
# The dataset will be downloaded if not present in the DATA_DIR.
train_dataset = CIFAR10(root=DATA_DIR, train=True, transform=train_transform, download=True)

# Load the CIFAR10 testing dataset with the defined testing transformation.
# The dataset will be downloaded if not present in the DATA_DIR.
test_set = CIFAR10(root=DATA_DIR, train=False, transform=test_transform, download=True)

# Split the training dataset into training and validation sets.
# The training set will have 45000 images, and the validation set will have 5000 images.
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
```

    Files already downloaded and verified
    Files already downloaded and verified


Let's setup the data loaders for training, validation and test datasets


```python
# Define the batch size for training, validation, and testing.
batch_size = 64

# Define the number of subprocesses to use for data loading.
num_workers = 16

# Create a DataLoader for the training and validation dataset:
# 1. Shuffle the training data for each epoch.
# 2. Drop the last batch if its size is not equal to `batch_size` to maintain consistency.
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers, 
                                           drop_last=True)

# Do not drop any data; process all the validation data.
val_loader = torch.utils.data.DataLoader(dataset=val_set, 
                                         batch_size=batch_size, 
                                         shuffle=False,
                                         num_workers=num_workers, 
                                         drop_last=False)

# Create a DataLoader for the testing dataset:
# Do not drop any data; process all the test data.
test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=num_workers, 
                                          drop_last=False)

```

Let's configure the model, optimization strategy, and training criterion.


```python
# Model, Loss and Optimizer
device = "cuda:0" if torch.cuda.is_available() else 0
args = ModelArgs()
model = VisionTransformer(args).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 130], gamma=0.1)
```

Time to bring our model to life! Let's train it.


```python
num_epochs = 150  # example value, adjust as needed

for epoch in range(num_epochs):
    
    # Training Phase
    model.train()
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_val_loss += loss.item()

            _, predicted = outputs.max(dim=-1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Update the learning rate
    lr_scheduler.step()

print("Training complete!")
```

    Epoch [1/150], Training Loss: 1.6899
    Epoch [1/150], Validation Loss: 1.4533, Validation Accuracy: 47.44%
    Epoch [2/150], Training Loss: 1.3940
    Epoch [2/150], Validation Loss: 1.3406, Validation Accuracy: 52.12%
    Epoch [3/150], Training Loss: 1.2890
    Epoch [3/150], Validation Loss: 1.2699, Validation Accuracy: 54.58%
    Epoch [4/150], Training Loss: 1.2265
    Epoch [4/150], Validation Loss: 1.2358, Validation Accuracy: 55.82%
    Epoch [5/150], Training Loss: 1.1703
    Epoch [5/150], Validation Loss: 1.1882, Validation Accuracy: 57.82%
    Epoch [6/150], Training Loss: 1.1259
    Epoch [6/150], Validation Loss: 1.1857, Validation Accuracy: 57.64%
    Epoch [7/150], Training Loss: 1.0941
    Epoch [7/150], Validation Loss: 1.1299, Validation Accuracy: 60.42%
    Epoch [8/150], Training Loss: 1.0640
    Epoch [8/150], Validation Loss: 1.1399, Validation Accuracy: 60.20%
    Epoch [9/150], Training Loss: 1.0369
    Epoch [9/150], Validation Loss: 1.0889, Validation Accuracy: 61.30%
    Epoch [10/150], Training Loss: 1.0141
    Epoch [10/150], Validation Loss: 1.0968, Validation Accuracy: 61.90%
    Epoch [11/150], Training Loss: 0.9868
    Epoch [11/150], Validation Loss: 1.0888, Validation Accuracy: 61.44%
    Epoch [12/150], Training Loss: 0.9644
    Epoch [12/150], Validation Loss: 1.0679, Validation Accuracy: 62.32%
    Epoch [13/150], Training Loss: 0.9433
    Epoch [13/150], Validation Loss: 1.0241, Validation Accuracy: 64.46%
    Epoch [14/150], Training Loss: 0.9187
    Epoch [14/150], Validation Loss: 1.0402, Validation Accuracy: 63.56%
    Epoch [15/150], Training Loss: 0.8963
    Epoch [15/150], Validation Loss: 0.9972, Validation Accuracy: 63.82%
    Epoch [16/150], Training Loss: 0.8820
    Epoch [16/150], Validation Loss: 1.0196, Validation Accuracy: 64.02%
    Epoch [17/150], Training Loss: 0.8663
    Epoch [17/150], Validation Loss: 0.9855, Validation Accuracy: 65.76%
    Epoch [18/150], Training Loss: 0.8485
    Epoch [18/150], Validation Loss: 1.0054, Validation Accuracy: 64.94%
    Epoch [19/150], Training Loss: 0.8254
    Epoch [19/150], Validation Loss: 0.9780, Validation Accuracy: 65.96%
    Epoch [20/150], Training Loss: 0.8090
    Epoch [20/150], Validation Loss: 0.9675, Validation Accuracy: 66.44%
    Epoch [21/150], Training Loss: 0.7890
    Epoch [21/150], Validation Loss: 1.0001, Validation Accuracy: 65.78%
    Epoch [22/150], Training Loss: 0.7722
    Epoch [22/150], Validation Loss: 0.9517, Validation Accuracy: 67.04%
    Epoch [23/150], Training Loss: 0.7527
    Epoch [23/150], Validation Loss: 0.9570, Validation Accuracy: 67.88%
    Epoch [24/150], Training Loss: 0.7401
    Epoch [24/150], Validation Loss: 0.9521, Validation Accuracy: 68.02%
    Epoch [25/150], Training Loss: 0.7248
    Epoch [25/150], Validation Loss: 0.9360, Validation Accuracy: 68.12%
    Epoch [26/150], Training Loss: 0.7117
    Epoch [26/150], Validation Loss: 0.9336, Validation Accuracy: 68.30%
    Epoch [27/150], Training Loss: 0.6987
    Epoch [27/150], Validation Loss: 0.9312, Validation Accuracy: 67.50%
    Epoch [28/150], Training Loss: 0.6816
    Epoch [28/150], Validation Loss: 0.9451, Validation Accuracy: 67.90%
    Epoch [29/150], Training Loss: 0.6623
    Epoch [29/150], Validation Loss: 0.9152, Validation Accuracy: 68.66%
    Epoch [30/150], Training Loss: 0.6503
    Epoch [30/150], Validation Loss: 0.9192, Validation Accuracy: 69.20%
    Epoch [31/150], Training Loss: 0.6428
    Epoch [31/150], Validation Loss: 0.9114, Validation Accuracy: 69.56%
    Epoch [32/150], Training Loss: 0.6296
    Epoch [32/150], Validation Loss: 0.9056, Validation Accuracy: 70.02%
    Epoch [33/150], Training Loss: 0.6153
    Epoch [33/150], Validation Loss: 0.9175, Validation Accuracy: 69.74%
    Epoch [34/150], Training Loss: 0.6040
    Epoch [34/150], Validation Loss: 0.9234, Validation Accuracy: 69.80%
    Epoch [35/150], Training Loss: 0.5926
    Epoch [35/150], Validation Loss: 0.9089, Validation Accuracy: 69.42%
    Epoch [36/150], Training Loss: 0.5833
    Epoch [36/150], Validation Loss: 0.8936, Validation Accuracy: 70.34%
    Epoch [37/150], Training Loss: 0.5667
    Epoch [37/150], Validation Loss: 0.9148, Validation Accuracy: 69.70%
    Epoch [38/150], Training Loss: 0.5603
    Epoch [38/150], Validation Loss: 0.9252, Validation Accuracy: 69.48%
    Epoch [39/150], Training Loss: 0.5485
    Epoch [39/150], Validation Loss: 0.9144, Validation Accuracy: 70.00%
    Epoch [40/150], Training Loss: 0.5410
    Epoch [40/150], Validation Loss: 0.9125, Validation Accuracy: 70.52%
    Epoch [41/150], Training Loss: 0.5261
    Epoch [41/150], Validation Loss: 0.9058, Validation Accuracy: 71.62%
    Epoch [42/150], Training Loss: 0.5160
    Epoch [42/150], Validation Loss: 0.9063, Validation Accuracy: 70.76%
    Epoch [43/150], Training Loss: 0.5077
    Epoch [43/150], Validation Loss: 0.8986, Validation Accuracy: 70.64%
    Epoch [44/150], Training Loss: 0.4996
    Epoch [44/150], Validation Loss: 0.8938, Validation Accuracy: 70.66%
    Epoch [45/150], Training Loss: 0.4871
    Epoch [45/150], Validation Loss: 0.9238, Validation Accuracy: 71.42%
    Epoch [46/150], Training Loss: 0.4801
    Epoch [46/150], Validation Loss: 0.9169, Validation Accuracy: 71.20%
    Epoch [47/150], Training Loss: 0.4713
    Epoch [47/150], Validation Loss: 0.9525, Validation Accuracy: 70.42%
    Epoch [48/150], Training Loss: 0.4653
    Epoch [48/150], Validation Loss: 0.9200, Validation Accuracy: 71.84%
    Epoch [49/150], Training Loss: 0.4452
    Epoch [49/150], Validation Loss: 0.9327, Validation Accuracy: 71.26%
    Epoch [50/150], Training Loss: 0.4470
    Epoch [50/150], Validation Loss: 0.9436, Validation Accuracy: 70.78%
    Epoch [51/150], Training Loss: 0.4336
    Epoch [51/150], Validation Loss: 0.9236, Validation Accuracy: 70.98%
    Epoch [52/150], Training Loss: 0.4281
    Epoch [52/150], Validation Loss: 0.9301, Validation Accuracy: 71.10%
    Epoch [53/150], Training Loss: 0.4190
    Epoch [53/150], Validation Loss: 0.9385, Validation Accuracy: 72.12%
    Epoch [54/150], Training Loss: 0.4148
    Epoch [54/150], Validation Loss: 0.9273, Validation Accuracy: 71.50%
    Epoch [55/150], Training Loss: 0.4019
    Epoch [55/150], Validation Loss: 0.9512, Validation Accuracy: 71.18%
    Epoch [56/150], Training Loss: 0.3956
    Epoch [56/150], Validation Loss: 0.9820, Validation Accuracy: 70.80%
    Epoch [57/150], Training Loss: 0.3975
    Epoch [57/150], Validation Loss: 0.9822, Validation Accuracy: 70.32%
    Epoch [58/150], Training Loss: 0.3845
    Epoch [58/150], Validation Loss: 0.9181, Validation Accuracy: 72.16%
    Epoch [59/150], Training Loss: 0.3780
    Epoch [59/150], Validation Loss: 0.9841, Validation Accuracy: 71.28%
    Epoch [60/150], Training Loss: 0.3689
    Epoch [60/150], Validation Loss: 0.9459, Validation Accuracy: 71.58%
    Epoch [61/150], Training Loss: 0.3699
    Epoch [61/150], Validation Loss: 0.9939, Validation Accuracy: 71.10%
    Epoch [62/150], Training Loss: 0.3594
    Epoch [62/150], Validation Loss: 0.9653, Validation Accuracy: 71.36%
    Epoch [63/150], Training Loss: 0.3531
    Epoch [63/150], Validation Loss: 0.9317, Validation Accuracy: 72.24%
    Epoch [64/150], Training Loss: 0.3504
    Epoch [64/150], Validation Loss: 0.9774, Validation Accuracy: 71.78%
    Epoch [65/150], Training Loss: 0.3368
    Epoch [65/150], Validation Loss: 0.9948, Validation Accuracy: 71.40%
    Epoch [66/150], Training Loss: 0.3347
    Epoch [66/150], Validation Loss: 1.0241, Validation Accuracy: 70.48%
    Epoch [67/150], Training Loss: 0.3301
    Epoch [67/150], Validation Loss: 1.0519, Validation Accuracy: 71.28%
    Epoch [68/150], Training Loss: 0.3236
    Epoch [68/150], Validation Loss: 1.0081, Validation Accuracy: 71.56%
    Epoch [69/150], Training Loss: 0.3199
    Epoch [69/150], Validation Loss: 1.0345, Validation Accuracy: 70.82%
    Epoch [70/150], Training Loss: 0.3181
    Epoch [70/150], Validation Loss: 1.0487, Validation Accuracy: 70.40%
    Epoch [71/150], Training Loss: 0.3109
    Epoch [71/150], Validation Loss: 1.0138, Validation Accuracy: 71.58%
    Epoch [72/150], Training Loss: 0.3036
    Epoch [72/150], Validation Loss: 0.9940, Validation Accuracy: 71.54%
    Epoch [73/150], Training Loss: 0.3001
    Epoch [73/150], Validation Loss: 0.9994, Validation Accuracy: 71.92%
    Epoch [74/150], Training Loss: 0.2936
    Epoch [74/150], Validation Loss: 1.0004, Validation Accuracy: 72.50%
    Epoch [75/150], Training Loss: 0.2865
    Epoch [75/150], Validation Loss: 1.0030, Validation Accuracy: 72.52%
    Epoch [76/150], Training Loss: 0.2870
    Epoch [76/150], Validation Loss: 1.0306, Validation Accuracy: 71.62%
    Epoch [77/150], Training Loss: 0.2810
    Epoch [77/150], Validation Loss: 1.0080, Validation Accuracy: 71.76%
    Epoch [78/150], Training Loss: 0.2727
    Epoch [78/150], Validation Loss: 1.0280, Validation Accuracy: 71.56%
    Epoch [79/150], Training Loss: 0.2795
    Epoch [79/150], Validation Loss: 1.0204, Validation Accuracy: 72.30%
    Epoch [80/150], Training Loss: 0.2698
    Epoch [80/150], Validation Loss: 1.0563, Validation Accuracy: 71.54%
    Epoch [81/150], Training Loss: 0.2002
    Epoch [81/150], Validation Loss: 0.9806, Validation Accuracy: 73.40%
    Epoch [82/150], Training Loss: 0.1714
    Epoch [82/150], Validation Loss: 0.9933, Validation Accuracy: 73.38%
    Epoch [83/150], Training Loss: 0.1620
    Epoch [83/150], Validation Loss: 1.0270, Validation Accuracy: 73.60%
    Epoch [84/150], Training Loss: 0.1577
    Epoch [84/150], Validation Loss: 1.0376, Validation Accuracy: 73.18%
    Epoch [85/150], Training Loss: 0.1501
    Epoch [85/150], Validation Loss: 1.0294, Validation Accuracy: 73.86%
    Epoch [86/150], Training Loss: 0.1451
    Epoch [86/150], Validation Loss: 1.0292, Validation Accuracy: 73.18%
    Epoch [87/150], Training Loss: 0.1413
    Epoch [87/150], Validation Loss: 1.0519, Validation Accuracy: 73.30%
    Epoch [88/150], Training Loss: 0.1352
    Epoch [88/150], Validation Loss: 1.0369, Validation Accuracy: 73.48%
    Epoch [89/150], Training Loss: 0.1329
    Epoch [89/150], Validation Loss: 1.0769, Validation Accuracy: 73.52%
    Epoch [90/150], Training Loss: 0.1285
    Epoch [90/150], Validation Loss: 1.0830, Validation Accuracy: 73.22%
    Epoch [91/150], Training Loss: 0.1302
    Epoch [91/150], Validation Loss: 1.0712, Validation Accuracy: 73.74%
    Epoch [92/150], Training Loss: 0.1246
    Epoch [92/150], Validation Loss: 1.0762, Validation Accuracy: 73.60%
    Epoch [93/150], Training Loss: 0.1223
    Epoch [93/150], Validation Loss: 1.0939, Validation Accuracy: 73.42%
    Epoch [94/150], Training Loss: 0.1240
    Epoch [94/150], Validation Loss: 1.1096, Validation Accuracy: 73.38%
    Epoch [95/150], Training Loss: 0.1139
    Epoch [95/150], Validation Loss: 1.0929, Validation Accuracy: 74.02%
    Epoch [96/150], Training Loss: 0.1173
    Epoch [96/150], Validation Loss: 1.0833, Validation Accuracy: 73.38%
    Epoch [97/150], Training Loss: 0.1148
    Epoch [97/150], Validation Loss: 1.1129, Validation Accuracy: 73.24%
    Epoch [98/150], Training Loss: 0.1140
    Epoch [98/150], Validation Loss: 1.0968, Validation Accuracy: 73.82%
    Epoch [99/150], Training Loss: 0.1082
    Epoch [99/150], Validation Loss: 1.0973, Validation Accuracy: 73.50%
    Epoch [100/150], Training Loss: 0.1059
    Epoch [100/150], Validation Loss: 1.1145, Validation Accuracy: 73.34%
    Epoch [101/150], Training Loss: 0.1098
    Epoch [101/150], Validation Loss: 1.1246, Validation Accuracy: 73.56%
    Epoch [102/150], Training Loss: 0.1064
    Epoch [102/150], Validation Loss: 1.1639, Validation Accuracy: 73.04%
    Epoch [103/150], Training Loss: 0.1046
    Epoch [103/150], Validation Loss: 1.1358, Validation Accuracy: 73.40%
    Epoch [104/150], Training Loss: 0.1038
    Epoch [104/150], Validation Loss: 1.1430, Validation Accuracy: 73.58%
    Epoch [105/150], Training Loss: 0.1005
    Epoch [105/150], Validation Loss: 1.1416, Validation Accuracy: 73.24%
    Epoch [106/150], Training Loss: 0.0983
    Epoch [106/150], Validation Loss: 1.1619, Validation Accuracy: 73.68%
    Epoch [107/150], Training Loss: 0.0998
    Epoch [107/150], Validation Loss: 1.1506, Validation Accuracy: 72.86%
    Epoch [108/150], Training Loss: 0.0961
    Epoch [108/150], Validation Loss: 1.1693, Validation Accuracy: 73.08%
    Epoch [109/150], Training Loss: 0.0960
    Epoch [109/150], Validation Loss: 1.1525, Validation Accuracy: 73.42%
    Epoch [110/150], Training Loss: 0.0952
    Epoch [110/150], Validation Loss: 1.1363, Validation Accuracy: 73.34%
    Epoch [111/150], Training Loss: 0.0938
    Epoch [111/150], Validation Loss: 1.1478, Validation Accuracy: 73.66%
    Epoch [112/150], Training Loss: 0.0937
    Epoch [112/150], Validation Loss: 1.1786, Validation Accuracy: 72.62%
    Epoch [113/150], Training Loss: 0.0960
    Epoch [113/150], Validation Loss: 1.1389, Validation Accuracy: 74.38%
    Epoch [114/150], Training Loss: 0.0963
    Epoch [114/150], Validation Loss: 1.1423, Validation Accuracy: 73.98%
    Epoch [115/150], Training Loss: 0.0920
    Epoch [115/150], Validation Loss: 1.2146, Validation Accuracy: 73.08%
    Epoch [116/150], Training Loss: 0.0907
    Epoch [116/150], Validation Loss: 1.1713, Validation Accuracy: 74.14%
    Epoch [117/150], Training Loss: 0.0881
    Epoch [117/150], Validation Loss: 1.1700, Validation Accuracy: 74.02%
    Epoch [118/150], Training Loss: 0.0885
    Epoch [118/150], Validation Loss: 1.2050, Validation Accuracy: 73.46%
    Epoch [119/150], Training Loss: 0.0873
    Epoch [119/150], Validation Loss: 1.1751, Validation Accuracy: 73.90%
    Epoch [120/150], Training Loss: 0.0871
    Epoch [120/150], Validation Loss: 1.2048, Validation Accuracy: 73.16%
    Epoch [121/150], Training Loss: 0.0857
    Epoch [121/150], Validation Loss: 1.1955, Validation Accuracy: 73.90%
    Epoch [122/150], Training Loss: 0.0888
    Epoch [122/150], Validation Loss: 1.1778, Validation Accuracy: 74.22%
    Epoch [123/150], Training Loss: 0.0855
    Epoch [123/150], Validation Loss: 1.2173, Validation Accuracy: 73.90%
    Epoch [124/150], Training Loss: 0.0855
    Epoch [124/150], Validation Loss: 1.2119, Validation Accuracy: 73.58%
    Epoch [125/150], Training Loss: 0.0850
    Epoch [125/150], Validation Loss: 1.2192, Validation Accuracy: 73.44%
    Epoch [126/150], Training Loss: 0.0843
    Epoch [126/150], Validation Loss: 1.2356, Validation Accuracy: 73.58%
    Epoch [127/150], Training Loss: 0.0839
    Epoch [127/150], Validation Loss: 1.2065, Validation Accuracy: 73.96%
    Epoch [128/150], Training Loss: 0.0804
    Epoch [128/150], Validation Loss: 1.1999, Validation Accuracy: 73.98%
    Epoch [129/150], Training Loss: 0.0794
    Epoch [129/150], Validation Loss: 1.2208, Validation Accuracy: 74.02%
    Epoch [130/150], Training Loss: 0.0791
    Epoch [130/150], Validation Loss: 1.2553, Validation Accuracy: 73.84%
    Epoch [131/150], Training Loss: 0.0794
    Epoch [131/150], Validation Loss: 1.2067, Validation Accuracy: 73.92%
    Epoch [132/150], Training Loss: 0.0776
    Epoch [132/150], Validation Loss: 1.1839, Validation Accuracy: 74.04%
    Epoch [133/150], Training Loss: 0.0764
    Epoch [133/150], Validation Loss: 1.2115, Validation Accuracy: 73.54%
    Epoch [134/150], Training Loss: 0.0748
    Epoch [134/150], Validation Loss: 1.1721, Validation Accuracy: 73.82%
    Epoch [135/150], Training Loss: 0.0733
    Epoch [135/150], Validation Loss: 1.1770, Validation Accuracy: 74.46%
    Epoch [136/150], Training Loss: 0.0740
    Epoch [136/150], Validation Loss: 1.2385, Validation Accuracy: 73.68%
    Epoch [137/150], Training Loss: 0.0741
    Epoch [137/150], Validation Loss: 1.1922, Validation Accuracy: 73.50%
    Epoch [138/150], Training Loss: 0.0743
    Epoch [138/150], Validation Loss: 1.2139, Validation Accuracy: 74.32%
    Epoch [139/150], Training Loss: 0.0744
    Epoch [139/150], Validation Loss: 1.2394, Validation Accuracy: 73.30%
    Epoch [140/150], Training Loss: 0.0733
    Epoch [140/150], Validation Loss: 1.2017, Validation Accuracy: 74.46%
    Epoch [141/150], Training Loss: 0.0748
    Epoch [141/150], Validation Loss: 1.2100, Validation Accuracy: 74.30%
    Epoch [142/150], Training Loss: 0.0745
    Epoch [142/150], Validation Loss: 1.2148, Validation Accuracy: 73.82%
    Epoch [143/150], Training Loss: 0.0702
    Epoch [143/150], Validation Loss: 1.1930, Validation Accuracy: 73.54%
    Epoch [144/150], Training Loss: 0.0741
    Epoch [144/150], Validation Loss: 1.2268, Validation Accuracy: 74.28%
    Epoch [145/150], Training Loss: 0.0711
    Epoch [145/150], Validation Loss: 1.2005, Validation Accuracy: 74.30%
    Epoch [146/150], Training Loss: 0.0721
    Epoch [146/150], Validation Loss: 1.2021, Validation Accuracy: 73.56%
    Epoch [147/150], Training Loss: 0.0692
    Epoch [147/150], Validation Loss: 1.2096, Validation Accuracy: 73.98%
    Epoch [148/150], Training Loss: 0.0691
    Epoch [148/150], Validation Loss: 1.2201, Validation Accuracy: 73.78%
    Epoch [149/150], Training Loss: 0.0696
    Epoch [149/150], Validation Loss: 1.2306, Validation Accuracy: 73.82%
    Epoch [150/150], Training Loss: 0.0704
    Epoch [150/150], Validation Loss: 1.2389, Validation Accuracy: 74.00%
    Training complete!



```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:  
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, predicted = outputs.max(dim=-1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

    Test Accuracy: 75.07%



```python

```
