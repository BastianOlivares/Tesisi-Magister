# RECORDATORIO

# - CREAR EWL DOCUMENTO RESULTADOS.TXT
# - REVISAR BIEN LAS DIRECCIONES DE LOS datasets
# - Y GREAR LA CARPTEA "OUTPUTGRAFICOS" PARA LAS IMAGENES DE LOS GR'AFICOS'

import torch
import torch.nn as nn # neural network
import torch.nn.functional as F # funcitions Relu
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

ITER = 2
EPOCH = 2
BATCH_SIZE = 32
LR = 0.01
MOMENTUM = 0.8


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Evaluación de la precisión en el conjunto de validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return train_losses, val_accuracies

def calculate_precision_recall(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calcula precisión y recall
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    
    # Calcula matriz de confusión
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return precision, recall, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names, i):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(f'OUTPUTGRAFICOS\matrix_confusion{i}.jpeg')
    #plt.show()


for i in range(ITER):
    print(f"-------------------------------------------- ITERACION: {i}")
    # ===========================
    # =======VERIFICAR GPU=======
    # ============================
    # Verificar si CUDA está disponible y qué dispositivo se está usando
    print("AREA DE GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        # Contar el número de GPUs disponibles
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Mostrar el nombre de cada GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
            print("CUDA is not available. Using CPU.")




    # =============================
    # =======LECTURA DATASET=======
    # =============================
    print("AREA DE DATASETS")
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carga las imágenes desde la carpeta --> TRAIN
    carpeta_imagenes_train = "./Datasets/malevis/malevis_train_val_224x224/train"
    dataset_train = datasets.ImageFolder(carpeta_imagenes_train, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_train

    # Carga las imágenes desde la carpeta --> TEST
    carpeta_imagenes_test = "./Datasets/malevis/malevis_train_val_224x224/val"
    dataset_test = datasets.ImageFolder(carpeta_imagenes_test, transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    #=====================
    #======MODELOS========
    #=====================
    print("AREA DE ELECCION DE MODELO")
    # Puedes cambiar el modelo según sea necesario:
    model = models.alexnet(pretrained=False)
    # model = models.vgg11(pretrained=False)
    # model = models.vgg16(pretrained=False)
    # model = models.resnet18(pretrained=False)
    # model = models.resnet34(pretrained=False)
    # model = models.resnet50(pretrained=False)
    # model = models.resnet101(pretrained=False)
    # model = models.inception_v3(pretrained=False)
    # model = models.densenet121(pretrained=False)
    # model = models.densenet169(pretrained=False)
    # model = models.densenet201(pretrained=False)

    # Enviar el modelo a la GPU (o a CPU si no hay GPU)
    model = model.to(device=device)

    # Configurar DataParallel si hay más de una GPU disponible
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)


    #========================
    #======PARAMETROS========
    #========================
    print("ELECCION DE PARAMETROS")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)


    #===========================
    #======ENTRENAMIENTO========
    #===========================
    print("ENTRENAMIENTO DEL MODELO")
    train_losses, val_accuracies = train_model( model, dataloader_train, dataloader_test, criterion, optimizer, device, num_epochs=EPOCH)

    #===============================================
    #======GRAFICOS RESULTADOS ENTRENAMIENTO========
    #===============================================
    print("GRAFICOS DEL ENTRENAMIENTO")

    # Graficar la pérdida de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(f'OUTPUTGRAFICOS\loss{i}.jpeg')
    #plt.show()

    # Graficar la precisión de validación
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig(f'OUTPUTGRAFICOS\precision{i}.jpeg')
    #plt.show()

    #========================================
    #======GRAFICOS RESULTADOS TESTEO========
    #========================================
    precision, recall, conf_matrix = calculate_precision_recall(model, dataloader_test, device)
    with open('resultados.txt', 'a') as archivo:
        archivo.write(f"""
        iteracion: {i}
            presicion {i} = {precision}
            recall {i}    = {recall}   
        """)
    class_names = dataset_test.classes
    plot_confusion_matrix(conf_matrix, class_names, i)