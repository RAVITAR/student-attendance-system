import torch
from mobilefacenet import MobileFaceNet

model = MobileFaceNet()
model.load_state_dict(torch.load('mobilefacenet.pt', map_location='cpu'))
model.eval()

# Save only the weights (state_dict)
torch.save(model.state_dict(), 'models/face_embedder.pth')
