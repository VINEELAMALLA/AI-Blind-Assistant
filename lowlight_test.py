import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import sys
sys.path.append(r"C:\Users\vinee\OneDrive\project-v\models\Zero_DCE\Zero_DCE_code")
from models.Zero_DCE.Zero_DCE_code import data_loader
import model
import Myloss
from torchvision import transforms
import cv2
def enhance_image(image):
    """Enhance the image for better low-light visibility."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available. Using CPU.")

    cudnn.benchmark = True  # Optimizes convolution performance
    
    DCE_net = model.enhance_net_nopool().to(device)  # Removed scale_factor
    
    try:
        DCE_net = torch.compile(DCE_net)
    except AttributeError:
        print("Warning: torch.compile not available. Skipping optimization.")

    if config.load_pretrain and os.path.exists(config.pretrain_dir):
        print("Loading pretrained model...")
        DCE_net.load_state_dict(torch.load(config.pretrain_dir, map_location=device))
    else:
        print(f"Pretrained model not found at {config.pretrain_dir}. Training from scratch.")

    train_dataset = data_loader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, 
        num_workers=config.num_workers, pin_memory=True
    )

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_TV = Myloss.L_TV()
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    DCE_net.train()
    accumulation_steps = 4  # Adjust based on available GPU memory

    cap = cv2.VideoCapture(0)
    
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break
            
            img_lowlight = img_lowlight.to(device)
            E = 0.6

            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                enhanced_image, A = DCE_net(img_lowlight)
                Loss_TV = config.loss_tv_multiplier * L_TV(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col = config.loss_color_multiplier * torch.mean(L_color(enhanced_image))
                loss_exp = config.loss_exp_multiplier * torch.mean(L_exp(enhanced_image, E))

                loss = Loss_TV + loss_spa + loss_col + loss_exp
            
            if scaler:
                scaler.scale(loss).backward()
                if (iteration + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            if (iteration + 1) % config.snapshot_iter == 0:
                torch.save(DCE_net.state_dict(), os.path.join(config.snapshots_folder, f"Epoch{epoch}.pth"))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting training...")
                cap.release()
                cv2.destroyAllWindows()
                return
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--loss_tv_multiplier', type=float, default=1600.0)
    parser.add_argument('--loss_color_multiplier', type=float, default=5.0)
    parser.add_argument('--loss_exp_multiplier', type=float, default=10.0)

    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Zero_DCE++/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots_Zero_DCE++/Epoch99.pth")
    
    config = parser.parse_args()
    os.makedirs(config.snapshots_folder, exist_ok=True)
    
    train(config)