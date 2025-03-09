import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt


def collate_fn(x):
    return x[0]


def get_image_embeddings(path, model, aligned):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # https://github.com/timesler/facenet-pytorch

    model = InceptionResnetV1(pretrained=model).eval().to(device)

    aligned = torch.stack(aligned).to(device)
    embeddings = model(aligned).detach().cpu()

    return embeddings    

def get_single_embedding(model, image):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Load a pre-trained InceptionResnetV1 model
    model = InceptionResnetV1(pretrained=model).eval().to(device)

    embedding = model(image).detach().cpu()

    return embedding


def tensor_to_image(tensor):
    """Convert a normalized tensor to a valid image array."""
    image = tensor.permute(1, 2, 0).detach().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype('uint8')
    return image