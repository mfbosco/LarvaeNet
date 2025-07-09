import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.patches import Patch
import cv2


# Função para checar proporção de targets
def check_targets(data):
    targets = [x for x in data if "000001_" in str(x)]
    print(f"Total: {len(data)}")
    print(f"Targets: {len(targets)}")
    print(f"Não Targets: {len(data) - len(targets)}")
    print(f"Proporção: {len(targets) / len(data)}")


### FUNÇÕES PARA SET UP DE TREINAMENTO E VALIDAÇÃO DOS MODELOS ###

# Funçao de critério de classificação com perda e regularização L2
def Criterion(model, preds, targets, device):
    ce = nn.CrossEntropyLoss().to(device)
    loss = ce(preds, targets.long())
    # add l2_regularization
    l2_regularization = 0
    for param in model.parameters():
        l2_regularization += torch.norm(param, 2)
    loss += 0.0001*l2_regularization  # 0.0001 is the weight_decay
    # compute mean accuracy in the batch
    pred_labels = torch.max(preds, 1)[1]  # same as argmax
    acc = torch.sum(pred_labels == targets.data)
    n = pred_labels.size(0)
    acc = acc/n
    kappa = cohen_kappa_score(pred_labels.cpu().numpy(),targets.data.cpu().numpy())
    return loss, acc, kappa

# Função para treinamento de modelo
def train_batch(model, data, optimizer, criterion, device):
    model.train()
    ims, targets = data
    ims = ims.to(device=device)
    targets = targets.to(device=device)
    preds = model(ims)
    loss, acc, kappa = criterion(model, preds, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), acc.item(), kappa

# Função para validação de modelo
@torch.no_grad()
def validate_batch(model, data, criterion, device):
    model.eval()
    ims, targets = data
    ims = ims.to(device=device)
    targets = targets.to(device=device)
    preds = model(ims)
    loss, acc, kappa = criterion(model, preds, targets)
    return loss.item(), acc.item(), kappa

# Função para visualização de resultados
def plot_results(log):
    # plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # plot loss
    log.plot_epochs(['trn_loss', 'val_loss'], ax=axes[0])
    axes[0].set_title('Loss', fontsize=14)

    # plot accuracy
    log.plot_epochs(['trn_acc', 'val_acc'], ax=axes[1])
    axes[1].set_title('Accuracy', fontsize=14)

    # plot kappa
    log.plot_epochs(['trn_kappa', 'val_kappa'], ax=axes[2])
    axes[2].set_title('Kappa', fontsize=14)

    plt.tight_layout()
    plt.show()

    return None

# Função para avaliação de modelo
def Test(model, testload, criterion, device):
    N = len(testload)
    mean_loss = 0
    mean_acc = 0
    mean_kappa = 0
    for bx, data in enumerate(testload):
        loss, acc, kappa = validate_batch(model, data, criterion, device)
        mean_loss += loss
        mean_acc += acc
        mean_kappa += kappa
    mean_loss = mean_loss / N
    mean_acc = mean_acc / N
    mean_kappa = mean_kappa / N
    return (mean_loss, mean_acc, mean_kappa)

#### FUNÇÕES PARA VIZUALIZAÇÃO DE FEATURE MAPS ####
# Função para extrair feature map da parte convolucional da rede 
def get_feature_map(model, input_tensor: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        feature_map = model.features(input_tensor) # extrai features da parte convolucional
    return feature_map

# Função para salvar imagens da camada convolucional
def save_feature_map_as_image(feature_map: torch.Tensor, outputdir: str) -> None:
    # remove batch dimension
    feature_map = feature_map[0] # shape: [C, H, W]
    
    # criando diretorio de saída se não existir
    os.makedirs(outputdir, exist_ok=True)

    for i in range(feature_map.shape[0]):
        fmap = feature_map[i].cpu().numpy()
        # normalize the feature map values to the range [0, 255]
        fmap_norm = 255 * (fmap - np.min(fmap)) / (np.max(fmap) - np.min(fmap))
        # convert the feature map to the uint8 data type
        fmap_norm = fmap_norm.astype(np.uint8)
        # save the feature map as an image
        img = Image.fromarray(fmap_norm)
        img.save(os.path.join(outputdir, f'feature_map_channel_{i}.png'))

# Função para visualizar feature maps
def visualize_features(feature_map: torch.Tensor,
                       max_channels: int = 16,
                       start_channel: int = 0,
                       figsize: tuple = (15, 10),
                       sample_idx: int = 0
                       ) -> None:
    """
    Visualize LarvaeNet feature maps

    Args:
        feature_map: Feature map tensor (batch_size, channels, height, width)
        max_channels: Maximum number of channels to display
        start_channel: Starting channel index (0-based)
        figsize: Figure size for matplotlib
        sample_idx: Which sample from the batch to visualize
    """
    # seleciona sample do batch, se for 4D (batch, canais, altura, largura)
    if feature_map.dim() == 4:
        # Shape: (channels, height, width)
        feature_map_sample = feature_map[sample_idx]
    else:
        feature_map_sample = feature_map

    # define total de canais e o intervalo de canais a serem visualizados
    total_channels = feature_map_sample.shape[0]
    end_channel = min(start_channel + max_channels, total_channels)
    num_channels = end_channel - start_channel

    if start_channel >= total_channels:
        raise ValueError(
            f"start_channel ({start_channel}) must be less than total channels ({total_channels})")

    print(f"Feature map shape: {feature_map.shape}")
    print(f"Total channels: {total_channels}")
    print(f"Visualizing channels {start_channel}-{end_channel-1} from sample {sample_idx}")

    # Calculate grid dimensions (max 4 col, rows definido de acordo)
    cols = 4
    rows = (num_channels + cols - 1) // cols

    # cria grid de canais
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # para cada canal, cria um subplot
    for i in range(num_channels):
        ax = axes[i]
        channel_idx = start_channel + i  # indice do canal
        # extrai canal, move para cpu e converte para numpy
        channel_data = feature_map_sample[channel_idx].cpu().numpy()

        # plota como imagem
        im = ax.imshow(channel_data, cmap='viridis')
        ax.set_title(f'Channel {channel_idx}\nMin: {channel_data.min():.2f}\nMax: {channel_data.max():.2f}',
                     fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)

    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'LarvaeNet Feature Maps - Channels {start_channel}-{end_channel-1}\nShape: {feature_map.shape}',
                 fontsize=14)
    plt.tight_layout()
    plt.show()


### FUNÇÕES PARA VIZUALIZAÇÃO DE PROJEÇÕES DE ATIVAÇÕES ###
# Função para pegar outpus de cada camada
def get_each_output(model, x):
    """Get the output of each layer of the model"""
    #empty dict
    output_by_layer = OrderedDict()
  
    #get the input
    output_by_layer['input'] = x.clone().detach().cpu().data.numpy()

    #for each layer of the feature extractor
    for layer_name, layer in model.features.named_children():
        #do forward through the layer   
        x = layer.forward(x)
        #save the output
        output_by_layer["features-"+layer_name] = x.clone().detach().cpu().numpy()

    x = torch.flatten(x, start_dim=1) # flatten the tensor
    output_by_layer['flattened'] = x.clone().detach().cpu().data.numpy()

    #for each layer of the classifier (note that you could have done that for model.conv1 and model.conv2 as well)
    for layer_name, layer in model.classifier.named_children():
        #do forward through the layer   
        x = layer.forward(x)
        #save the output
        output_by_layer["classifier-"+layer_name] = x.clone().detach().cpu().numpy()
  
    #return output by layer
    return output_by_layer

# Função para pegar outputs de ultima camada conv e ultima camada densa
def get_outputs_by_layer(model, x):
    """Get only last conv layer and last dense layer (classifier)"""
    #empty dict
    output_by_layer = OrderedDict()
  
    #get the input
    output_by_layer['input'] = x.clone().detach().cpu().data.numpy()

    #get only feature extractor
    features = model.features(x)
    output_by_layer['features'] = features.clone().detach().cpu().data.numpy()

    x = torch.flatten(x, start_dim=1) # flatten the tensor
    output_by_layer['flattened'] = x.clone().detach().cpu().data.numpy()

    #get only classifier
    classifier = model.classifier(x)
    output_by_layer['classifier'] = classifier.clone().detach().cpu().data.numpy()
  
    #return output by layer
    return output_by_layer

# Função para pegar outputs e labels
def get_ouputs(model, dataload, device, get_outputs_function):
    outputs_by_layer = None
    all_labels = None

    #get a batch from the dataload
    for inputs, labels in dataload:
        #move inputs to the correct device
        inputs = inputs.to(device)
        labels = labels.clone().detach().cpu().numpy()

        #get the activations for visualization
        outputs = get_outputs_function(model, inputs) # one of the function's defined above

        #save the outputs
        if outputs_by_layer is None:
            outputs_by_layer = outputs
            all_labels       = labels
        else:
            for layer in outputs:
                outputs_by_layer[layer] = np.concatenate((outputs_by_layer[layer], outputs[layer]), axis=0)
            all_labels = np.concatenate((all_labels, labels))   

    return outputs_by_layer, all_labels

# Função para projetar ativações em 2D
def projection(outputs_by_layer, reducer):
    projection_by_layer = OrderedDict()

    for layer in outputs_by_layer:
        #get the output of layer
        output = outputs_by_layer[layer]
        output = output.reshape(output.shape[0], -1)
        #map to 2D
        embedded = reducer.fit_transform(output)
        #save projection
        projection_by_layer[layer] = embedded
  
    return projection_by_layer

# Funcão para visualizar projeções de ativações
def create_visualization(projection_by_layer_umap, projection_by_layer_tsne, all_labels):
    
    colors = ['steelblue' if label == 0 else 'firebrick' for label in all_labels]
    
    for layer in projection_by_layer_umap:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # UMAP plot
        embedded = projection_by_layer_umap[layer]
        scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1], c=colors, alpha=0.7)
        axes[0].set_title(f'{layer} - UMAP')
        
        # Criar legenda manual
        legend_elements = [Patch(facecolor='steelblue', label='Impurities (0)'),
                           Patch(facecolor='firebrick', label='Larvae (1)')]
        axes[0].legend(handles=legend_elements)
        
        # t-SNE plot  
        embedded2 = projection_by_layer_tsne[layer]
        scatter2 = axes[1].scatter(embedded2[:, 0], embedded2[:, 1], c=colors, alpha=0.7)
        axes[1].set_title(f'{layer} - t-SNE')
        axes[1].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)


### FUNÇÕES PARA VIZUALIZAÇÃO DE REGIÕES DE ATENÇÃO (GradCAM) ###
# Função para gerar GradCAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_image, class_idx=None):
        model_output = self.model(input_image)
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1)
        self.model.zero_grad()
        class_score = model_output[:, class_idx]
        class_score.backward()
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=[2, 3])
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(input_image.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        return cam.detach().cpu().numpy()

# Função para gerar GradCAM para um modelo customizado
def gradcam_custom_model(model, image, target_layer_name):
    target_layer = dict(model.named_modules())[target_layer_name]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(image)
    return cam

# Função para visualizar GradCAM
def visualize_gradcam(orig_image, cam, alpha=0.4):
    cam_resized = cv2.resize(cam, (orig_image.shape[1], orig_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = heatmap * alpha + orig_image * (1 - alpha)
    overlayed = 255*(overlayed - np.min(overlayed)) / (np.max(overlayed)-np.min(overlayed) + 1e-8)
    overlayed = overlayed.astype('uint8')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed.astype(np.uint8))
    axes[2].set_title('Overlayed Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

    return overlayed, heatmap
