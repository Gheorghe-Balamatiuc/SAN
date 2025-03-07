import argparse
import os
import random
import tempfile
import time
import numpy as np
import ray
import torch

from dataset import get_dataset
from models.Backbone import Backbone
from utils import load_config
from ray.train import Checkpoint
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from utils import updata_lr, Meter, cal_score
from tqdm import tqdm


def training(params, model, optimizer, epoch, train_loader):
    
    model.train()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    for batch_idx, (images, image_masks, labels, label_masks) in enumerate(train_loader):

        images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
            device), label_masks.to(device)

        batch, time = labels.shape[:2]
        if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
            updata_lr(optimizer, epoch, batch_idx, len(train_loader), params['epoches'], params['lr'])
        optimizer.zero_grad()

        probs, loss = model(images, image_masks, labels, label_masks)

        word_loss, struct_loss, parent_loss, kl_loss = loss
        loss = (word_loss + struct_loss + parent_loss + kl_loss)

        loss.backward()
        if params['gradient_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])

        optimizer.step()

        loss_meter.add(loss.item())

        wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

        word_right = word_right + wordRate * time
        struct_right = struct_right + structRate * time
        exp_right = exp_right + ExpRate * batch
        length = length + time
        cal_num = cal_num + batch

        if batch_idx % 10 == 9:
            print(f'Epoch: {epoch+1} batch index: {batch_idx+1}/{len(train_loader)} train loss: {loss.item():.4f} word loss: {word_loss:.4f} '
                                 f'struct loss: {struct_loss:.4f} parent loss: {parent_loss:.4f} '
                                 f'kl loss: {kl_loss:.4f} WordRate: {word_right / length:.4f} '
                                 f'structRate: {struct_right / length:.4f} ExpRate: {exp_right / cal_num:.4f}')

    return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader):

    model.eval()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    for batch_idx, (images, image_masks, labels, label_masks) in enumerate(eval_loader):

        images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
            device), label_masks.to(device)

        batch, time = labels.shape[:2]

        probs, loss = model(images, image_masks, labels, label_masks, is_train=False)

        word_loss, struct_loss = loss
        loss = word_loss + struct_loss
        loss_meter.add(loss.item())

        wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

        word_right = word_right + wordRate * time
        struct_right = struct_right + structRate * time
        exp_right = exp_right + ExpRate
        length = length + time
        cal_num = cal_num + batch

    return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num

def train_tune(config, **kwargs):

    """config"""
    params = load_config(kwargs['args'].config)

    base_dir = "/workspaces/Anaconda/SAN"
    params.update(config)

    """random seed"""
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    # Construct absolute paths
    params['word_path'] = os.path.join(base_dir, params['word_path'])
    params['train_image_path'] = os.path.join(base_dir, params['train_image_path'])
    params['train_label_path'] = os.path.join(base_dir, params['train_label_path'])
    params['eval_image_path'] = os.path.join(base_dir, params['eval_image_path'])
    params['eval_label_path'] = os.path.join(base_dir, params['eval_label_path'])
    params['checkpoint_dir'] = os.path.join(base_dir, params['checkpoint_dir'])

    train_loader, eval_loader = get_dataset(params)
    model = Backbone(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
                f'max_size-{params["image_height"]}-{params["image_width"]}'
    print(model.name)
    model = model.to(device)

    optimizer_cls = getattr(torch.optim, params['optimizer'])
    if params['optimizer'] == "SGD":
        optimizer = optimizer_cls(
            model.parameters(),
            lr=float(params['lr']),
            weight_decay=float(params['weight_decay'])
        )
    else:
        optimizer = optimizer_cls(
            model.parameters(),
            lr=float(params['lr']),
            eps=float(params['eps']),
            weight_decay=float(params['weight_decay'])
        )
    
    # Load existing checkpoint through `get_checkpoint()` API.
    start = 0
    checkpoint = train.get_checkpoint()
    print('checkpoint:', checkpoint)
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state, optimizer_state, prev_epoch = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt"),
                map_location=params["device"]
            )
            print('Checkpoint size: ', os.path.getsize(os.path.join(checkpoint_dir, "checkpoint.pt")) / 1024 / 1024, 'MB')
            start = prev_epoch + 1
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            print('Start from epoch:', start)
    
    min_score = 0
    min_step = 0
    for epoch in range(start, params['epoches']):
        
        train_loss, train_word_score, train_node_score, train_expRate = training(params, model, optimizer, epoch, train_loader)

        eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader)

        print(f'Epoch: {epoch+1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
              f'ExpRate: {eval_expRate:.4f}')
        
        if eval_expRate > min_score and not kwargs['args'].check:
            min_score = eval_expRate
            min_step = 0
        
        elif min_score != 0 and 'lr_decay' in params and params['lr_decay'] == 'step':

            min_step += 1

            if min_step > params['step_ratio']:
                new_lr = optimizer.param_groups[0]['lr'] / params['step_decay']

                if new_lr < params['lr'] / 1000:
                    print('lr is too small')
                    exit(-1)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                min_step = 0
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict(), epoch), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            print('loss:', eval_loss, 'accuracy:', eval_expRate)
            train.report(
                {"loss": eval_loss, "accuracy": eval_expRate},
                checkpoint=checkpoint,
            )
            print('checkpoint done')
    
    print("Finished Training")


def main(num_samples=20, max_num_epochs=20, gpus_per_trial=1):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    parser.add_argument('--check', action='store_true', help='only for code check')
    args = parser.parse_args()

    if not args.config:
        print('please provide config yaml')
        exit(-1)

    config = {
        "batch_size": tune.choice([2, 4, 8]),
        "optimizer": tune.choice(['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']),
        "lr": tune.loguniform(1e-4, 1),
        "lr_decay": tune.choice(['step', 'cosine']),
        "step_ratio": tune.choice([5, 10, 20, 30, 40]),
        "weight_decay": tune.loguniform(1e-4, 1),
        "dropout": tune.choice([True, False]),
        "dropout_ratio": tune.uniform(0.2, 0.5),
        "relu": tune.choice([True, False]),
        "gradient": tune.choice([0.1, 1, 5, 10, 100]),
        "gradient_clip": tune.choice([True, False]),
        "use_label_mask": tune.choice([True, False]),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    initial_params = [
        {
            "batch_size": 8,
            "optimizer": 'Adadelta',
            "lr": 1,
            "lr_decay": 'cosine',
            "step_ratio": 10,
            "weight_decay": 1e-4,
            "dropout": True,
            "dropout_ratio": 0.5,
            "relu": True,
            "gradient": 100,
            "gradient_clip": True,
            "use_label_mask": False,
        },
    ]
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=1)

    storage_path = os.path.expanduser("~/ray_results")
    exp_name = "tune"
    path = os.path.join(storage_path, exp_name)

    if tune.Tuner.can_restore(path):
        tuner = tune.Tuner.restore(
            path, 
            tune.with_resources(
                tune.with_parameters(train_tune, args=args),
                resources={"cpu": 16, "gpu": gpus_per_trial}
            ),
            resume_errored=True,
            param_space=config,
        )
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_tune, args=args),
                resources={"cpu": 16, "gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric="accuracy",
                mode="min",
                scheduler=scheduler,
                search_alg=algo,
                num_samples=num_samples,
            ),
            param_space=config,
            run_config=RunConfig(storage_path=storage_path, name=exp_name),
        )
    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["accuracy"]))


if __name__ == "__main__":
    main()