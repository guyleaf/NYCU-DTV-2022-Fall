import pandas as pd
import torch
import torch.nn.functional as F
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Network
from utils import make_dataset


class ArgumentParser(Tap):
    data_folder: str = "data"  # the root of data folder
    output_csv: str = "HW1_311553007.csv"
    model: str = "model.pth"
    batch_size: int = 100  # the size of mini-batch
    num_workers: int = 0


@torch.no_grad()
def test(
    model: Network, dataloader: DataLoader, device: str = "cpu"
) -> torch.Tensor:
    model.eval()

    results = []
    for imgs, _ in tqdm(dataloader, desc="Test batch", leave=False):
        imgs = imgs.to(device)
        pred_logits: torch.Tensor = model(imgs)
        pred_labels = F.softmax(pred_logits, dim=1)
        results.append(torch.argmax(pred_labels, dim=1).cpu())

    return torch.concat(results, dim=0)


def main(args: ArgumentParser):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = make_dataset(args.data_folder, "test")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Network(3, 10)
    weights = torch.load(args.model)["model"]
    model.load_state_dict(weights)

    model.to(device)
    pred_labels = test(model, test_dataloader, device)

    csv_content = pd.DataFrame(
        data=dict(names=test_dataset._img_names, label=pred_labels.numpy())
    )
    csv_content.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
