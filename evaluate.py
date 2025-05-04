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
import os


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
            print(f"saving!!! {config.csv_path.split('/')[-1].split('.')[-2]}")

            import cv2
            import matplotlib.pyplot as plt
            import os

            valid_mask = (depth >= config.min_depth) & (depth <= config.max_depth)

            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            # img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            pred_temp = nn.functional.interpolate(
                    pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            print(f"pred_temp shape: {pred_temp.shape}")
            pred_np = pred_temp.squeeze().cpu().numpy()
            depth_np = depth.squeeze().cpu().numpy()
            # depth_np[depth_np == 0] = np.nan
            
            valid_pred = pred_temp[valid_mask].cpu().numpy()
            valid_depth = depth[valid_mask].cpu().numpy()
            print(f"pred depth: min: {np.min(valid_pred)}, max: {np.max(valid_pred)}")
            print(f"depth_np: min: {np.min(valid_depth)}, max: {np.max(valid_depth)}")
            

            # Calculate error only on valid regions
            error_values = np.abs(valid_pred - valid_depth)
            cmap = plt.get_cmap("turbo_r")
            norm_error_values = (error_values - np.min(error_values)) / (np.max(error_values) - np.min(error_values))
            color_norm_error_values = (cmap(norm_error_values)[..., :3] * 255).astype(np.uint8)
            
            # error_map = np.full_like(pred_np, fill_value=60)  
            error_map = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)

            print(f"valid_mask shape: {valid_mask.squeeze().cpu().numpy().shape}")
            print(f"error_map shape: {error_map.shape}")

            error_map[valid_mask.squeeze().cpu().numpy()] = (0.8 * color_norm_error_values + (1 - 0.8) * error_map[valid_mask.squeeze().cpu().numpy()]).astype(np.uint8)  
            print(f"valid_mask_shape: {np.sum(valid_mask.cpu().numpy())}, error_values shape: {error_values.shape}")

            min_error = np.nanmin(error_values)
            max_error = np.nanmax(error_values)
            print(f"min_error: {min_error}; max_error: {max_error}")

            
            # Create output dir
            os.makedirs(f"/mnt/GrandTour/visualizations/{config.csv_path.split('/')[-1].split('.')[-2]}", exist_ok=True) # args.dataset_file_path.split('/')[-1].split('.')[-2]
                        
            fig, axes = plt.subplots(2, 3, figsize=(36, 20))
            fig.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing between subplots

            # First row
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')

            im = axes[0, 1].imshow(error_map, cmap='turbo_r', vmin=min_error, vmax=max_error)
            fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[0, 1].set_title("Error Map")
            axes[0, 1].axis('off')

            norm_pred_np = (pred_np - np.min(pred_np)) / (np.max(pred_np) - np.min(pred_np))
            im = axes[0, 2].imshow(norm_pred_np, cmap='turbo_r', vmin=np.min(norm_pred_np), vmax=np.max(norm_pred_np))
            fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[0, 2].set_title("Normalized Predicted Depth")
            axes[0, 2].axis('off')

            # Second row
            im = axes[1, 0].imshow(pred_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[1, 0].set_title("Predicted Depth")
            axes[1, 0].axis('off')

            im = axes[1, 1].imshow(depth_np, cmap='turbo_r', vmin=np.min(depth_np), vmax=np.max(depth_np))
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Depth (m)')
            axes[1, 1].set_title("GT Depth")
            axes[1, 1].axis('off')

            norm_depth_np = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
            cmap = plt.get_cmap('turbo_r')
            colored_depth = cmap(norm_depth_np)[..., :3] 

            # Alpha blend with RGB image
            overlay_img = np.copy(img_np)
            overlay_img[valid_mask.squeeze().cpu().numpy()] = colored_depth[valid_mask.squeeze().cpu().numpy()]

            axes[1, 2].imshow(overlay_img)
            axes[1, 2].set_title("GT Depth Overlay")
            axes[1, 2].axis('off')
            

            base_vis_dir = f"/mnt/GrandTour/visualizations/{config.csv_path.split('/')[-1].split('.')[-2]}"
            os.makedirs(base_vis_dir, exist_ok=True)
            image_path = sample['image_path'][0]
            print(f"image_path: {image_path}")
            timestamp = image_path.split()[0].split('/')[-1].split('.')[0]

            visuals = {
                "original_image": (img_np, None, "Original Image"),
                "error_map": (error_map, (min_error, max_error), "Error Map"),
                "normalized_pred_depth": (norm_pred_np, (np.min(norm_pred_np), np.max(norm_pred_np)), "Normalized Predicted Depth"),
                "predicted_depth": (pred_np, (np.min(depth_np), np.max(depth_np)), "Predicted Depth"),
                "gt_depth": (depth_np, (np.min(depth_np), np.max(depth_np)), "GT Depth"),
                "gt_overlay": (overlay_img, None, "GT Depth Overlay"),
            }

            for key, (data, vrange, title) in visuals.items():
                fig, ax = plt.subplots(figsize=(12, 6))
                if vrange:
                    im = ax.imshow(data, cmap='turbo_r', vmin=vrange[0], vmax=vrange[1])
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Depth (m)')
                else:
                    ax.imshow(data)
                ax.set_title(title)
                ax.axis('off')

                out_dir = os.path.join(base_vis_dir, key)
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, f"{timestamp}.png"))
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
