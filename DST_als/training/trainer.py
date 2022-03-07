import torch


class Trainer:
    """

    """

    def __init__(self, model, device):
        self.device = device
        self.model = model.to(device)

        self.optimizer = self.config_optimizer()
        self.scheduler = self.config_scheduler(self.optimizer)
        self.loss = self.config_loss()
        metrics = self.config_metrics()
        if not isinstance(metrics, list):
            metrics = [metrics]
        self.metrics = metrics

    def fit(self,
            train_data,
            dev_data,
            callbacks=None,
            epochs=30):
        model = self.to_device(self.model)
        model.stop_training = False
        validation = val_y is not None

        if validation:
            validation = True
            val_g = g if val_g is None else val_g
            val_data = self.config_test_data(val_g, val_y, val_index)

        # Setup callbacks
        self.callbacks = callbacks = self.config_callbacks(verbose, epochs, callbacks=callbacks)
        train_data = self.config_train_data(g, y, index)

        logs = BunchDict()

        if verbose:
            print("Training...")

        callbacks.on_train_begin()
        try:
            for epoch in range(epochs):
                callbacks.on_epoch_begin(epoch)
                train_logs = self.train_step(train_data)
                logs.update({k: self.to_item(v) for k, v in train_logs.items()})

                if validation:
                    valid_logs = self.test_step(val_data)
                    logs.update({("val_" + k): self.to_item(v) for k, v in valid_logs.items()})

                callbacks.on_epoch_end(epoch, logs)

                if model.stop_training:
                    print(f"Early Stopping at Epoch {epoch}", file=sys.stderr)
                    break

        finally:
            callbacks.on_train_end()

        return self