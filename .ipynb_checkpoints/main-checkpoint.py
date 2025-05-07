#!/usr/bin/env python3
# ------------------------------------------------------------
#  PyTorch-CIFAR  |  multi-model • multi-optimiser benchmark
# ------------------------------------------------------------
import argparse, math, time, pathlib, traceback
from collections import defaultdict

import torch, torch.nn as nn, torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision, torchvision.transforms as transforms
import pandas as pd, matplotlib.pyplot as plt

# repo-local imports
from models import *
from utils  import progress_bar
from optim.sgd import DAG           # our Dynamic-AlphaGrad optimiser
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--lr",      default=0.1,  type=float)
parser.add_argument("--epochs",  default=50,  type=int)
parser.add_argument("--batch",   default=128,  type=int)
parser.add_argument("--workers", default=2,    type=int)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------
print("==> Preparing CIFAR-10")
tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])
tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10("./data", True,  download=True, transform=tf_train),
    batch_size=args.batch, shuffle=True,  num_workers=args.workers)
testloader  = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10("./data", False, download=True, transform=tf_test),
    batch_size=100, shuffle=False, num_workers=args.workers)
# ------------------------------------------------------------
MODEL_CATALOG = [
    ("VGG19",            lambda: VGG("VGG19")),
    ("ResNet18",         ResNet18),
    ("PreActResNet18",   PreActResNet18),
    ("GoogLeNet",        GoogLeNet),
    ("DenseNet121",      DenseNet121),
    ("ResNeXt29",        ResNeXt29_2x64d),
    ("MobileNet",        MobileNet),
    ("MobileNetV2",      MobileNetV2),
    ("DPN92",            DPN92),
    ("ShuffleNetG2",     ShuffleNetG2),
    ("SENet18",          SENet18),
    ("ShuffleNetV2",     lambda: ShuffleNetV2(1)),
    ("EfficientNetB0",   EfficientNetB0),
    ("RegNetX_200MF",    RegNetX_200MF),
    ("SimpleDLA",        SimpleDLA),
    ("DLA",              DLA),
]
# ------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
COLORS    = dict(SGD="tab:blue", Adam="tab:orange", DAG="tab:green")

# ------------------------------------------------------------
def train_epoch(net, opt):
    net.train()
    loss_sum, correct, total = 0., 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out   = net(x)
        loss  = criterion(out, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item()
        correct  += out.argmax(1).eq(y).sum().item()
        total    += y.size(0)

        progress_bar(i, len(trainloader),
                     f"Loss:{loss_sum/(i+1):.3f} | Acc:{100.*correct/total:.2f}%")
    return loss_sum/len(trainloader), 100.*correct/total, (time.time()-t0)/len(trainloader)


@torch.no_grad()
def test_epoch(net):
    net.eval()
    loss_sum, correct, total = 0., 0, 0
    for i, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        out  = net(x)
        loss_sum += criterion(out, y).item()
        correct  += out.argmax(1).eq(y).sum().item()
        total    += y.size(0)

        progress_bar(i, len(testloader),
                     f"Loss:{loss_sum/(i+1):.3f} | Acc:{100.*correct/total:.2f}%")
    return loss_sum/len(testloader), 100.*correct/total
# ------------------------------------------------------------
RESULTS_DIR = pathlib.Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# cosine-decay helper for k_val
def k_decay(step, total, k0=1.5, k1=0.8):
    """cosine decay from k0 → k1 over <total> epochs"""
    return k1 + 0.5*(k0-k1)*(1+math.cos(math.pi*step/total))
# ------------------------------------------------------------
for model_name, ctor in MODEL_CATALOG:
    print(f"\n\n############  {model_name}  ############")
    try:
        net = ctor().to(device)
        if device == "cuda":
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        opt_cfgs = {
            # Uncomment if you also want the baselines:
            "SGD":  dict(cls=optim.SGD,  lr=args.lr, momentum=0.9, weight_decay=5e-4),
            "Adam": dict(cls=optim.Adam, lr=1e-3,   weight_decay=5e-4),
            "DAG":  dict(
                cls=DAG,
                lr=3e-5, momentum=0.9, k_val=2,
                weight_decay=5e-3,
                shrink=dict(s_min=0.05, lambda_rms=0.25),
                k_sched=DAG.cosine_decay(k0=2, total_steps=args.epochs),
            ),
        }

        hist   = defaultdict(lambda: defaultdict(list))
        outdir = RESULTS_DIR / model_name
        outdir.mkdir(parents=True, exist_ok=True)

        # ─── optimiser loop ───
        for opt_name, cfg in opt_cfgs.items():
            print(f"\n→ Optimiser: {opt_name}")

            # fresh weights
            net.apply(lambda m: m.reset_parameters()
                      if hasattr(m, "reset_parameters") else None)

            opt   = cfg["cls"](net.parameters(),
                               **{k: v for k, v in cfg.items() if k != "cls"})
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            for ep in range(args.epochs):
                print(f"\nEpoch {ep:03d}")

                if isinstance(opt, DAG):
                    opt.set_k_val(k_decay(ep, args.epochs))

                tr_loss, tr_acc, step_sec = train_epoch(net, opt)
                ts_loss, ts_acc           = test_epoch(net)

                # gather saturation (DAG only)
                if isinstance(opt, DAG):
                    sat_vals = [
                        st["sat_ratio"].item()
                        for st in opt.state.values()
                        if isinstance(st, dict) and "sat_ratio" in st
                    ]
                    opt._last_sat = sat_vals

                sched.step()

                h = hist[opt_name]
                h["train_loss"].append(tr_loss)
                h["test_loss"].append(ts_loss)
                h["train_acc"].append(tr_acc)
                h["test_acc"].append(ts_acc)
                h["step_time"].append(step_sec)
                h["gap"].append(ts_loss - tr_loss)

                if isinstance(opt, DAG) and getattr(opt, "_last_sat", []):
                    h["sat_ratio"].append(sum(opt._last_sat) / len(opt._last_sat))
                else:
                    h["sat_ratio"].append(float("nan"))

                print(f"[{opt_name}] ep{ep:03d} "
                      f"tr_acc={tr_acc:5.1f}% ts_acc={ts_acc:5.1f}% "
                      f"Δ={ts_loss - tr_loss:+.3f}  step={step_sec*1e3:6.1f} ms")

            pd.DataFrame(hist[opt_name]).to_csv(outdir / f"{opt_name}_metrics.csv",
                                                index=False)

        # ─── combined plots ───
        def plot_metric(kind, ylabel, train_key, test_key=None, logy=False):
            plt.figure(figsize=(6, 4))
            for opt, metrics in hist.items():
                x = range(len(metrics[train_key]))
                plt.plot(x, metrics[train_key],
                         color=COLORS[opt], linestyle='-',
                         label=f"{opt}-train")
                if test_key:
                    plt.plot(x, metrics[test_key],
                             color=COLORS[opt], linestyle='--',
                             label=f"{opt}-test")

            if logy:
                plt.yscale("log")

            plt.title(f"{model_name} – {ylabel}")
            plt.xlabel("epoch")
            plt.ylabel(ylabel)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=.3)
            plt.tight_layout()
            plt.savefig(outdir / f"{kind}.png")
            plt.close()

        plot_metric("loss",      "Cross-entropy",
                    train_key="train_loss", test_key="test_loss")
        plot_metric("accuracy",  "Accuracy (%)",
                    train_key="train_acc",  test_key="test_acc")
        plot_metric("step_time", "sec / step",
                    train_key="step_time")
        plot_metric("sat_ratio", "sat-ratio",
                    train_key="sat_ratio")

    except Exception as e:
        print(f"[!]   {model_name} failed: {e}")
        traceback.print_exc()
        continue