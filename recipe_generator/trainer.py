from tqdm import tqdm
import pickle

import torch

from recipe_generator.utils import nostdout, data_to_device

class Trainer():

    def __init__(self, train_cfg, train_dl, validation_dl, model, optimiser, loss_func, metric_funcs, sample_inference, save_dir, wandb, model_tgt=False):
        self.train_cfg = train_cfg
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.model = model
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.metric_funcs = metric_funcs
        self.sample_inference = sample_inference
        self.save_dir = save_dir
        self.wandb = wandb
        self.epoch = 0
        self.global_step = 0
        self.model_output = None
        self.metrics = {}
        self.early_stopper = EarlyStopping()
        self.validation_loss = float('inf')
        self.device = train_cfg.device
        # self.model_artifact = self.wandb.Artifact('model', type='model') if self.wandb else None

    def train(self):

        self.model.train()
        
        for epoch in range(self.train_cfg.n_epochs):

            self.epoch = epoch

            for i, batch in enumerate(tqdm(self.train_dl)):

                batch = data_to_device(batch, self.device)
                train_loss = self.step(batch)
                self.global_step += 1

                if i % self.train_cfg.save_steps == 0:
                    batch = next(iter(self.validation_dl))
                    batch = data_to_device(batch, self.device)
                    self.validation_loss = self.eval(batch)
                    self.metrics = self.calculate_metrics(train_loss, self.validation_loss, batch)
                    if self.train_cfg.wandb: self.wandb.log(self.metrics, step=self.global_step)

            self.save(batch)
            self.early_stopper(self.validation_loss)
            if self.early_stopper.early_stop: break

    def step(self, batch):

        self.model.train()

        xb, yb, mask = batch

        self.optimiser.zero_grad(set_to_none=True)
        self.model_output = self.model(xb)
        loss = self.loss_func(self.model_output, yb, mask)
        loss.backward()
        self.optimiser.step()

        return loss


    def eval(self, batch):

        self.model.eval()

        with torch.no_grad():

            xb, yb, mask = batch

            self.model_output = self.model(xb)
            loss = self.loss_func(self.model_output, yb, mask)

            return loss
        
    def calculate_metrics(self, train_loss, validation_loss, batch):

        metrics = {
            'epoch': self.epoch,
            'train_loss': train_loss,
            'validation_loss': validation_loss
        }

        for metric in self.metric_funcs:
            metrics.update({
                metric.__name__: metric(self, batch)
            })

        return metrics

    def save(self, batch):

        with open(self.save_dir/'metrics.pickle', 'wb') as f: pickle.dump(self.metrics, f)

        # if self.wandb: 
        #     with nostdout(): torch.onnx.export(self.model, batch, './outputs/weights/model.onnx')
        #     self.wandb.save('./outputs/weights/model.onnx')

        sample_outputs = self.sample_inference(self, batch)
        sample_outputs.to_string(self.save_dir/f'sample_inference{self.epoch}.txt')
        if self.wandb: self.wandb.save(self.save_dir/f'sample_inference{self.epoch}.txt')

        self.model.to('cpu')
        torch.save(self.model.state_dict(), self.save_dir/'model.pt')
        if self.wandb:
            artifact = self.wandb.Artifact('model', type='model')
            artifact.add_file(self.save_dir/'model.pt')
            self.wandb.log_artifact(artifact)
        self.model.to(self.train_cfg.device)

class EarlyStopping:
    def __init__(self, patience=2, verbose=False, sigma=0.01, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            sigma (float): Minimum ratio change in the validation loss to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.sigma = sigma
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < (self.best_score*(1+self.sigma)):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        

