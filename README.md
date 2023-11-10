# Coding Project: Reproducing Salehi et al. with ViT instead of VGG

This repository is a personal project in which I reproduce [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/pdf/2011.11108.pdf) and exchange the VGG backbone to a ViT backbone. The core idea of the original paper is to distill the knowledge of a pretrained VGG network that scores almost perfectly on the data to a smaller to-be-trained VGG network. This to-be-trained VGG is optimised to detect anomalies. In this project I don't focus as much on the loss function as my goal is just to implement a ViT in a research setting. Furthermore the goal of this project is to complete it within a week. Because of the time constraint I limit myself to reproducing and extending only on the CIFAR10 dataset. One other major constraint is that I did not have acces to a GPU cluster for this project, so all work had to be done on google colab. 

### Result base model (VGG)
Training the base model took extremely long (2.5 hours) and due to colab I could not run things in the background / in parallel, thus I did only one run/class. The result of that run can be found in vgg_kdad.ipynb ,  but in summary I trained for class 3, 200 epochs, 1e-3 learning rate, batch size 64. I got an RocAUC of 77.03%, which is almost exactly what was reported in the original paper (77.02%), thus it seemed reasonable to not investigate much further. One thing to note is that the original code-base used the same set for validation and testing, so this is essentially the validation RocAUC. 

### Result my model (VIT)
Because I wanted to work with pytorch lightning instead of base pytorch and because of convenience I decided to start from scratch. This lead to a stand-alone notebook vit_kdad.ipynb (i.e. it can be run without this repository). For the frozen pretrained ViT network I used 'nateraw/vit-base-patch16-224-cifar10' that can be found on Huggingface. This constrained the to-be-trained ViT as the embedding dimension and sequence length (i.e. number of patches) had to be the same for the loss function and the anomaly detection. For the rest I tried to keep the ViT as small as possible as it was overfitting quite quickly. The results per class can be found in the table below, and are compared to the base model. It can be seen that the model vastly outperforms the base model, and also the model was much faster to run. Below the table you can see the training loss and the validation RocAUC during training.

| Class | Validation RocAUC VGG (%) | Validation RocAUC ViT (%) | Test RocAUC ViT (%) |
|-------|-----------------------|-----------------------|-----------------|
| 0     | 90.53                 | 97.08                 | 96.78           |
| 1     | 90.35                 | 98.47                 | 98.16           |
| 2     | 79.66                 | 95.80                 | 96.20           |
| 3     | 77.02                 | 93.82                 | 93.55           |
| 4     | 86.71                 | 96.74                 | 96.52           |
| 5     | 91.4                  | 95.36                 | 94.72           |
| 6     | 88.98                 | 98.27                 | 97.96           |
| 7     | 86.78                 | 97.92                 | 97.71           |
| 8     | 91.45                 | 97.64                 | 98.01           |
| 9     | 88.91                 | 98.47                 | 98.68           |
| **Mean**  | **87.18**                | **96.96**             | **96.83**       |

<p float="left">
  <img src="images\train_loss.png" alt="First Image" width="500"/>
  <img src="images\val_roc_auc.png" alt="Second Image" width="500"/>
</p>

### Daily Log

**Monday (5 hours):** Started reading the paper and tried running the code. Because of colab I could not adhere to the package versions so I had to do some bug fixing for that. End of the day I managed to run the code so that was pretty smooth.

**Tuesday (8 hours):** Eventhough I could run the code the night before, the checkpoints were not yet working as the paths were hard-coded. Therefor I decided to restructure the code-base and I fixed the paths. I also started reading into ViTs as I had never really done so before. Also looked for a pretrained ViT on CIFAR10, which I found on Huggingface, and tried to run it.

**Wednesday (8 hours):** Started making my own load data function, because the pretrained ViT was trained on resized CIFAR10 and because the original code-base used the same set for validation and testing. This was suprisingly hard, also to get it working with the pretrained ViT. In the afternoon I started to implement the to-be-trained ViT in pytorch lightning. 

**Thursday (8 hours):** Started of by having to come back to my dataloader as it was not working as intended afterall. Then finished implementing the ViT, and spend a lot of time on fixing bugs and trying to understand the detection_tes() function in the original code base. End of day I actually get very good results, but still only for little epochs trained and only for normal_class=3.

**Friday (6 hours):** Started by incorporating wandb logging. Then, I started tuning for class 3 and when I finished with that I ran those hyperparameters also for the other classes. While that was running I did some reporting and cleaning of the code-base. I also briefly tried what happened if I used a pretrained ViT that was not fine tuned on CIFAR10, but the results were worse so I left it.




