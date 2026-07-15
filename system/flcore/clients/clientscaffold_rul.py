import torch
import numpy as np
import time
from flcore.clients.clientbase_rul import Client_RUL
from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer


class clientSCAFFOLD_RUL(Client_RUL):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)

        # self.model.cpu()
        self.num_batches = len(trainloader)
        self.update_yc(max_local_epochs)
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
            
        
    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    # ------------------------------------------------------------------ #
    # DEVIAZIONE DALLE REGOLE D'INGAGGIO (approvata dall'umano 2026-07-15)
    #
    # Questo file (clients/*_rul.py) e' nella lista "MUST NOT modify". La patch e'
    # stata autorizzata esplicitamente perche' il codice originale ha un bug che rende
    # SCAFFOLD non eseguibile sul task E, non un problema di iperparametri.
    #
    # Bug: num_batches = len(trainloader) con drop_last=True (clientbase_rul.py:37).
    # Sul task E (few-shot) ogni client ha 1 sola finestra < batch_size -> 0 batch ->
    #   1/self.num_batches = 1/0 -> nan/crash. Verificato: 10/10 client E con 0 batch.
    # Sul task C (21 finestre) -> 1 batch -> nessun crash.
    #
    # `nb = max(num_batches, 1)` evita la divisione per zero. ATTENZIONE alla semantica:
    # con 0 batch il loop di training in train() non fa NESSUN passo di ottimizzazione,
    # quindi il client non aggiorna il modello e il suo control-variate resta invariato.
    # La patch fa girare SCAFFOLD su E, ma quei client non contribuiscono: e' il minimo
    # per rendere il metodo eseguibile, NON un modo per renderlo competitivo sul few-shot.
    # Nessun effetto sui task con dati sufficienti: se num_batches>=1, il valore e' identico.
    # ------------------------------------------------------------------ #
    def update_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        nb = max(self.num_batches, 1)   # patch: evita 1/0 sui client few-shot (task E)
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/nb/max_local_epochs/self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        nb = max(self.num_batches, 1)   # patch: evita 1/0 sui client few-shot (task E)
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/nb/max_local_epochs/self.learning_rate * (x - yi))

        return delta_y, delta_c

