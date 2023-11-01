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
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x
```


```python
all_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
desired_classes = all_classes[0:8]
desired_indices = [all_classes.index(cls) for cls in desired_classes]
desired_indices
```




    [0, 1, 2, 3, 4, 5, 6, 7]




```python
DATA_DIR="../data"
def get_cifar10_data_loader():
    """
    Get the CIFAR10 data loader
    """
    test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                         ])


    # get the training and testing datasets
    train_dataset = CIFAR10(root=DATA_DIR, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=DATA_DIR, train=False, transform=test_transform, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
    # _, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    return train_set, val_set, test_dataset
```


```python
train_set, val_set, test_set = get_cifar10_data_loader()
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
def filter_dataset(dataset):
    filtered_indices = [i  for i,  (_, label) in enumerate(dataset) if label in desired_indices]
    return torch.utils.data.Subset(dataset, filtered_indices)
```


```python
train_set = filter_dataset(train_set)
val_set = filter_dataset(val_set)
test_set = filter_dataset(test_set)
print(f"len of train set {len(train_set)} val set {len(val_set)} test set {len(test_set)}")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 train_set = filter_dataset(train_set)
          2 val_set = filter_dataset(val_set)
          3 test_set = filter_dataset(test_set)


    NameError: name 'filter_dataset' is not defined



```python
@dataclass
class ModelArgs:
    dim:int =  256
    hidden_dim:int = 512
    n_heads:int = 8
    n_layers:int = 6
    patch_size:int = 4
    n_channels = 3
    n_patches = 64
    n_classes = 10
    dropout = 0.2
```


```python
class MultiHeadAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads*self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads*self.head_dim, self.dim, bias=False)
    
    def forward(self, x):
        b, seq_len, dim = x.shape
        
        assert dim == self.dim, "dim is not matching"
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
        k = k.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
        v = v.contiguous().view(b, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1,2)
        
        attn = torch.matmul(q, k. transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = F.softmax(attn, dim = -1)
        
        out = torch.matmul(attn_scores, v)
        out = out.contiguous().view(b, seq_len, -1)
        
        return self.wo(out)        
```


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


```python
class VisionTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.patch_size = args.patch_size
        
        self.input_layer = nn.Linear(args.n_channels * (args.patch_size ** 2), args.dim)
        attn_blocks = []
        for _ in range(args.n_layers):
            attn_blocks.append(AttentionBlock(args))
        
        self.transformer = nn.Sequential(*attn_blocks)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.n_classes)
        )
        
        self.dropout = nn.Dropout(args.dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1+args.n_patches, args.dim))
    
    def forward(self, x):
        x = img_to_patch(x, self.patch_size)
        b, seq_len, _ = x.shape
        x = self.input_layer(x)
        
        cls_token = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = x + self.pos_embedding[:,:seq_len+1]
        
        x = self.dropout(x)
        x = self.transformer(x)
        # print("========== x shape =====", x.shape)
        x = x.transpose(0, 1)
        cls = x[0]
        out = self.mlp(cls)
        return out
```


```python
args = ModelArgs()
args.dim
```




    256




```python
# Model, Loss and Optimizer
device = "cuda:0" if torch.cuda.is_available() else 0
args = ModelArgs()
model = VisionTransformer(args).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
```


```python
batch_size=64
num_workers = 16
# get the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers, drop_last=False)
```


```python
num_epochs = 50  # example value, adjust as needed

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
        # print("==== outputs shape ===", outputs.shape)
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
        for inputs, labels in val_loader:  # Assuming val_loader is defined elsewhere
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

    Epoch [1/50], Training Loss: 1.6999
    Epoch [1/50], Validation Loss: 1.5045, Validation Accuracy: 45.68%
    Epoch [2/50], Training Loss: 1.4183
    Epoch [2/50], Validation Loss: 1.3412, Validation Accuracy: 51.46%
    Epoch [3/50], Training Loss: 1.3052
    Epoch [3/50], Validation Loss: 1.2789, Validation Accuracy: 54.12%
    Epoch [4/50], Training Loss: 1.2441
    Epoch [4/50], Validation Loss: 1.1980, Validation Accuracy: 56.50%
    Epoch [5/50], Training Loss: 1.1810
    Epoch [5/50], Validation Loss: 1.1757, Validation Accuracy: 57.78%
    Epoch [6/50], Training Loss: 1.1459
    Epoch [6/50], Validation Loss: 1.1456, Validation Accuracy: 59.66%
    Epoch [7/50], Training Loss: 1.1104
    Epoch [7/50], Validation Loss: 1.1171, Validation Accuracy: 60.24%
    Epoch [8/50], Training Loss: 1.0766
    Epoch [8/50], Validation Loss: 1.1127, Validation Accuracy: 60.40%
    Epoch [9/50], Training Loss: 1.0458
    Epoch [9/50], Validation Loss: 1.0516, Validation Accuracy: 63.28%
    Epoch [10/50], Training Loss: 1.0224
    Epoch [10/50], Validation Loss: 1.0277, Validation Accuracy: 63.30%
    Epoch [11/50], Training Loss: 0.9897
    Epoch [11/50], Validation Loss: 1.0764, Validation Accuracy: 62.40%
    Epoch [12/50], Training Loss: 0.9665
    Epoch [12/50], Validation Loss: 0.9885, Validation Accuracy: 65.40%
    Epoch [13/50], Training Loss: 0.9433
    Epoch [13/50], Validation Loss: 1.0050, Validation Accuracy: 64.72%
    Epoch [14/50], Training Loss: 0.9206
    Epoch [14/50], Validation Loss: 0.9721, Validation Accuracy: 65.60%
    Epoch [15/50], Training Loss: 0.8898
    Epoch [15/50], Validation Loss: 0.9529, Validation Accuracy: 66.24%
    Epoch [16/50], Training Loss: 0.8695
    Epoch [16/50], Validation Loss: 0.9544, Validation Accuracy: 67.14%
    Epoch [17/50], Training Loss: 0.8494
    Epoch [17/50], Validation Loss: 0.9397, Validation Accuracy: 67.06%
    Epoch [18/50], Training Loss: 0.8323
    Epoch [18/50], Validation Loss: 0.9327, Validation Accuracy: 67.90%
    Epoch [19/50], Training Loss: 0.8093
    Epoch [19/50], Validation Loss: 0.9189, Validation Accuracy: 67.96%
    Epoch [20/50], Training Loss: 0.7944
    Epoch [20/50], Validation Loss: 0.9323, Validation Accuracy: 67.48%
    Epoch [21/50], Training Loss: 0.7782
    Epoch [21/50], Validation Loss: 0.8998, Validation Accuracy: 68.76%
    Epoch [22/50], Training Loss: 0.7570
    Epoch [22/50], Validation Loss: 0.9224, Validation Accuracy: 67.96%
    Epoch [23/50], Training Loss: 0.7479
    Epoch [23/50], Validation Loss: 0.8879, Validation Accuracy: 69.04%
    Epoch [24/50], Training Loss: 0.7303
    Epoch [24/50], Validation Loss: 0.8992, Validation Accuracy: 68.88%
    Epoch [25/50], Training Loss: 0.7172
    Epoch [25/50], Validation Loss: 0.9031, Validation Accuracy: 68.68%
    Epoch [26/50], Training Loss: 0.7002
    Epoch [26/50], Validation Loss: 0.9011, Validation Accuracy: 68.94%
    Epoch [27/50], Training Loss: 0.6896
    Epoch [27/50], Validation Loss: 0.8620, Validation Accuracy: 70.24%
    Epoch [28/50], Training Loss: 0.6741
    Epoch [28/50], Validation Loss: 0.8784, Validation Accuracy: 69.68%
    Epoch [29/50], Training Loss: 0.6650
    Epoch [29/50], Validation Loss: 0.8822, Validation Accuracy: 69.12%
    Epoch [30/50], Training Loss: 0.6537
    Epoch [30/50], Validation Loss: 0.8605, Validation Accuracy: 70.66%
    Epoch [31/50], Training Loss: 0.6316
    Epoch [31/50], Validation Loss: 0.8665, Validation Accuracy: 69.68%
    Epoch [32/50], Training Loss: 0.6229
    Epoch [32/50], Validation Loss: 0.8829, Validation Accuracy: 70.56%
    Epoch [33/50], Training Loss: 0.6127
    Epoch [33/50], Validation Loss: 0.9040, Validation Accuracy: 69.98%
    Epoch [34/50], Training Loss: 0.6039
    Epoch [34/50], Validation Loss: 0.8693, Validation Accuracy: 71.82%
    Epoch [35/50], Training Loss: 0.5903
    Epoch [35/50], Validation Loss: 0.8985, Validation Accuracy: 70.36%
    Epoch [36/50], Training Loss: 0.5815
    Epoch [36/50], Validation Loss: 0.8742, Validation Accuracy: 70.58%
    Epoch [37/50], Training Loss: 0.5719
    Epoch [37/50], Validation Loss: 0.8965, Validation Accuracy: 70.52%
    Epoch [38/50], Training Loss: 0.5587
    Epoch [38/50], Validation Loss: 0.9256, Validation Accuracy: 69.62%
    Epoch [39/50], Training Loss: 0.5453
    Epoch [39/50], Validation Loss: 0.8924, Validation Accuracy: 70.58%
    Epoch [40/50], Training Loss: 0.5360
    Epoch [40/50], Validation Loss: 0.8924, Validation Accuracy: 71.16%
    Epoch [41/50], Training Loss: 0.5312
    Epoch [41/50], Validation Loss: 0.8469, Validation Accuracy: 71.50%
    Epoch [42/50], Training Loss: 0.5159
    Epoch [42/50], Validation Loss: 0.8710, Validation Accuracy: 71.36%
    Epoch [43/50], Training Loss: 0.5002
    Epoch [43/50], Validation Loss: 0.8875, Validation Accuracy: 71.82%
    Epoch [44/50], Training Loss: 0.4954
    Epoch [44/50], Validation Loss: 0.8929, Validation Accuracy: 72.16%
    Epoch [45/50], Training Loss: 0.4857
    Epoch [45/50], Validation Loss: 0.9040, Validation Accuracy: 71.42%
    Epoch [46/50], Training Loss: 0.4764
    Epoch [46/50], Validation Loss: 0.9254, Validation Accuracy: 71.64%
    Epoch [47/50], Training Loss: 0.4638
    Epoch [47/50], Validation Loss: 0.8767, Validation Accuracy: 72.06%
    Epoch [48/50], Training Loss: 0.4673
    Epoch [48/50], Validation Loss: 0.8786, Validation Accuracy: 72.22%
    Epoch [49/50], Training Loss: 0.4463
    Epoch [49/50], Validation Loss: 0.9310, Validation Accuracy: 71.18%
    Epoch [50/50], Training Loss: 0.4459
    Epoch [50/50], Validation Loss: 0.8743, Validation Accuracy: 72.82%
    Training complete!



```python
def test_accuracy():
    # Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:  # Assuming val_loader is defined elsewhere
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            

            _, predicted = outputs.max(dim=-1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
```


```python
test_accuracy()
```

    Test Accuracy: 71.70%



```python
6.3857e+00
```




    6.3857




```python
1.3028e+01
```




    13.028




```python

```
