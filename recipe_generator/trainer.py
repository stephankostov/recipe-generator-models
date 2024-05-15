from tqdm import tqdm
import pickle

import torch

from recipe_generator.utils import nostdout

class Trainer():

    def __init__(self, train_cfg, train_dl, validation_dl, model, optimiser, loss_func, metric_funcs, sample_inference, save_dir, wandb):
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
        # self.model_artifact = self.wandb.Artifact('model', type='model') if self.wandb else None

    def train(self):

        self.model.train()
        
        for epoch in range(self.train_cfg.n_epochs):

            self.epoch = epoch

            for i, batch in enumerate(tqdm(self.train_dl)):

                train_loss = self.step(batch)
                self.global_step += 1

                if i % self.train_cfg.save_steps == 0:
                    batch = next(iter(self.validation_dl))
                    batch = [x.to(self.train_cfg.device) for x in batch]
                    validation_loss = self.eval(batch)
                    self.metrics = self.calculate_metrics(train_loss, validation_loss, batch)
                    if self.train_cfg.wandb: self.wandb.log(self.metrics, step=self.global_step)

            self.save(batch)

    def step(self, batch):

        self.model.train()

        batch = [x.to(self.train_cfg.device) for x in batch]
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

            batch = [x.to(self.train_cfg.device) for x in batch]
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

        

