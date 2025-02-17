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
from training import train, eval
from utils import load_config
from ray.train import Checkpoint
from ray import tune
from ray.tune.schedulers import ASHAScheduler


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
parser.add_argument('--check', action='store_true', help='only for code check')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)

"""config"""
params = load_config(args.config)

def train_tune(config, base_dir):

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

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))
    
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')
    
    # Load existing checkpoint through `get_checkpoint()` API.
    if ray.train.get_checkpoint():
        loaded_checkpoint = ray.train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    
    min_score = 0
    min_step = 0
    for epoch in range(params['epoches']):
        
        train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader)

        eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader)

        print(f'Epoch: {epoch+1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
              f'ExpRate: {eval_expRate:.4f}')
        
        if eval_expRate > min_score and not args.check:
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
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.train.report(
                {"loss": eval_loss, "accuracy": eval_expRate},
                checkpoint=checkpoint,
            )
    
    print("Finished Training")


def main(num_samples=20, max_num_epochs=10, gpus_per_trial=1):
    config = {
        "batch_size": tune.choice([2, 4, 8]),
        "optimizer": tune.choice(['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']),
        "lr": tune.loguniform(1e-4, 1),
        "lr_decay": tune.choice(['step', 'cosine']),
        "step_ratio": tune.choice([5, 10, 20, 30, 40]),
        "weight_decay": tune.loguniform(1e-4, 1),
        "dropout": tune.choice(['True', 'False']),
        "dropout_ratio": tune.uniform(0.2, 0.5),
        "relu": tune.choice(['True', 'False']),
        "gradient": tune.choice([0.1, 1, 5, 10, 100]),
        "gradient_clip": tune.choice(['True', 'False']),
        "use_label_mask": tune.choice(['True', 'False']),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_tune, base_dir="/workspaces/Anaconda/SAN"),
            resources={"cpu": 8, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(best_result.metrics["accuracy"]))


if __name__ == "__main__":
    main()