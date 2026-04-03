import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config_path, output_path, cp_name):
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"No such path exists: {output_path}")


    cp_dir_path = os.path.join(output_path, 'checkpoints')
    os.makedirs(cp_dir_path, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'val_accuracy': val_acc,
        'config_path': config_path
    }

    cp_path = os.path.join(cp_dir_path, f'{cp_name}.pth')
    torch.save(checkpoint, cp_path)
    print(f'Checkpoint "{cp_name}" saved at epoch {epoch} → {cp_path}')


def load_checkpoint(cp_path, model, optimizer=None, scheduler=None, device='cpu'):
    if not os.path.exists(cp_path):
        raise FileNotFoundError(f"No such path exists: {cp_path}")

    cp = torch.load(cp_path, map_location=device)

    model.load_state_dict(cp['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(cp['optimizer_state_dict'])

    if scheduler and cp['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(cp['scheduler_state_dict'])

    epoch = cp['epoch']
    val_acc = cp['val_accuracy']
    config_path = cp['config_path']

    return model, optimizer, scheduler, epoch + 1, val_acc, config_path