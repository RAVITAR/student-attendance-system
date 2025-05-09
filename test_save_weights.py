import torch
from mobilefacenet import MobileFaceNet

# Build model and load pretrained weights
model = MobileFaceNet()
model.load_state_dict(torch.load("mobilefacenet.pt", map_location="cpu"))
model.eval()

# ✅ Save only the weights
torch.save(model.state_dict(), "models/face_embedder.pth")
print("✅ Fresh state_dict saved to models/face_embedder.pth")
