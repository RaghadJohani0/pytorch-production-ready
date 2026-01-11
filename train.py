import torch
from PIL import Image
from torchvision import transforms
from src.models.classifier import ImageClassifier

def predict(image_path, checkpoint, num_classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ImageClassifier(num_classes)
    model.load_state_dict(torch.load(checkpoint)["model_state"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        return torch.argmax(output, dim=1).item()
