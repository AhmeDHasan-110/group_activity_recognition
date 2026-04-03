import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, dir_path: str, name: str):
        """
        Initializes a logger to save training information for a baseline.

        Args:
            dir_path (str): Directory path where log file is saved.
            name (str): name of the baseline for which the logger is used.
        """
        os.makedirs(dir_path, exist_ok=True)

        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(dir_path, f'{name}_train_{time}.log')

        self.logger = logging.getLogger(f'{name}_logger_{time}')
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()

        fmt = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.propagate = False
    
    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

def log_curves(curves: list, names: list, save_dir: str, file_name: str = '') -> str:
    """Create and save plots of training curves"""

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    
    ax[0].plot(curves[0], label=names[0])
    ax[0].plot(curves[1], label=names[1])
    if len(curves[0]) <= 20:
        ax[0].set_xticks(range(len(curves[0])))
    ax[0].grid()
    ax[0].legend()
    
    ax[1].plot(curves[2], label=names[2])
    ax[1].plot(curves[3], label=names[3])
    if len(curves[2]) <= 20:
        ax[1].set_xticks(range(len(curves[2])))
    ax[1].grid()
    ax[1].legend()

    name = f"{file_name}_training_curves.png" if file_name else 'training_curves.png'
    fig.savefig(os.path.join(save_dir, name))
    plt.show()
    plt.close(fig)

    return 'Training curves saved successfully.'
