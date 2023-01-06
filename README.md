# EVA8

## Session 2.5 PyTorch 101

### Task: 

![Alt text](https://user-images.githubusercontent.com/5630870/211112238-b6512297-f5e5-4103-be3c-15577b09cfc4.png)


### Input Description:

1. Generating MNIST images:  I have downloaded them from torchvision. 

```python 

   self.data = torchvision.datasets.MNIST('/content/mnist', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
```        
2. Generating Random numbers: I have created a 1D torch tensor. With a random number, I used eq() to get 1D True/False values and converted them to long int.

```python 

randomNumber = torch.tensor([0,1,2,3,4,5,6,7,8,9])
self.random = torch.randint(0, 10, (len(self.data),))
randomInput = randomNumber.eq(self.random[index]).long()
```

The Dataset implementation as follows: 

```python 

class TrainDataSet(Dataset):
    def __init__(self):
        self.data = torchvision.datasets.MNIST('/content/mnist', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        self.random = torch.randint(0, 10, (len(self.data),))
        
    def __getitem__(self, index):
        image, label = self.data[index]
        randomInput = randomNumber.eq(self.random[index]).long()
        return image, label, randomInput

    def __len__(self):
        return len(self.data)
```
        
        
