import argparse 
from pathlib import Path 
import torch 
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.nn.functional import cross_entropy 
from tqdm import tqdm 

from dataset import Flickr30kDataset
from model import MultimodalDecoder


