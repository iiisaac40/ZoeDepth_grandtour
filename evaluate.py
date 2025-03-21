# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import argparse
from pprint import pprint

import torch
import torch.nn
import torch.nn as nn
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np
import csv


from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    model.eval()
    metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=sample['dataset'][0], focal=focal)

        if i % 10 == 0 and 'vis_res' in config and config.vis_res == 'TRUE':  # Visualize every 10th sample
            import cv2
            import matplotlib.pyplot as plt
            import os

            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            pred_np = nn.functional.interpolate(
                    pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            pred_np = pred_np[0].cpu().numpy()
            depth_np = depth[0].cpu().numpy()

            pred_np = pred_np.squeeze()
            depth_np = depth_np.squeeze()

            valid_mask_np = (depth_np > 0) & (depth_np < 60)

            error_map = np.abs(pred_np - depth_np)
            min_error, max_error = np.min(error_map), np.max(error_map)
            error_map[~valid_mask_np] = np.nan 
            
            print(f"pred depth: min: {np.min(pred_np)}, max: {np.max(pred_np)}, {pred_np.shape}")
            print(f"depth_np: min: {np.min(depth_np)}, max: {np.max(depth_np)}, {depth_np.shape}")
                    
            # Create output dir
            os.makedirs("/home/output/visualizations/GrandTour_ZoeDepth_snow", exist_ok=True)
                        
            # Add prediction visualization to the plot
            plt.figure(figsize=(30, 20))
            
            # Original Image
            plt.subplot(221)
            plt.imshow(img_np)
            plt.title("Original Image")
            
            # Error Map
            plt.subplot(222)
            plt.imshow(error_map, cmap='turbo_r', vmin=min_error, vmax=max_error)
            plt.colorbar(label='Depth (m)')
            plt.title("Error Map")
            
            # Prediction
            plt.subplot(223)
            plt.imshow(pred_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            plt.colorbar(label='Depth (m)')
            plt.title("Predicted Depth")

            # GT depth
            plt.subplot(224)
            plt.imshow(depth_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            plt.colorbar(label='Depth (m)')
            plt.title("GT Depth")
            
            plt.savefig(f"/home/output/visualizations/GrandTour_ZoeDepth_snow/sample_{i}.png")
            plt.close()

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))



        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics

def main(config):
    # model = build_model(config)
    model = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True) if 'pretrained_resource' in config and '_K.pt' in config.pretrained_resource else torch.hub.load(".", "ZoeD_NK", source="local", pretrained=True)
    test_loader = DepthDataLoader(config, 'online_eval').data
    model = model.cuda()
    metrics = evaluate(model, test_loader, config)
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"

    csv_file = config.csv_path
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if f.tell() == 0:
            writer.writeheader() 
        writer.writerow(metrics)

    return metrics


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default=None, help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
