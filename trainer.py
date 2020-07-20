import torch as t
from sklearn.metrics import f1_score, accuracy_score
#from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
    
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._model.zero_grad()
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model.forward(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss
            
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        self._model.zero_grad()
        pred = self._model.forward(x)
        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss, pred
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        total_loss = 0
        for x, y in self._train_dl:
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss = self.train_step(x, y)
            total_loss += loss.item()
        # calculate the average loss for the epoch and return it
        total_loss = total_loss / len(self._train_dl)
        print("train: loss: {}".format(total_loss))
        return total_loss

    def val_test(self):
        # set eval mode
        self._model.eval()
        # disable gradient computation (disable autograd engine)
        t.no_grad()
        # iterate through the validation set
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        for x, y in self._val_test_dl:
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a validation step
            loss, pred = self.val_test_step(x, y)
            # calculate metrices for this iteration
            total_loss += loss.item()
            total_acc += accuracy_score(y.cpu(), pred.cpu() > 0.5)
            total_f1 += f1_score(y.cpu(), pred.cpu() > 0.5, average=None)
            # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        total_loss = total_loss / len(self._val_test_dl)
        total_acc = total_acc / len(self._val_test_dl)
        total_f1 = total_f1 / len(self._val_test_dl)
        # return the loss and print the calculated metrics
        print("test: loss: {}, accuracy: {}%, f-score: {}".format(total_loss, total_acc, total_f1))
        t.enable_grad()
        return total_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses = []
        test_losses = []
        epoch = 0
        
        last_test_loss = float('inf')
        stop_iterations = 0
        while True:
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            test_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if epoch % 10 == 0:
                # store checkpoint every 10 epochs
                self.save_checkpoint(epoch)
            epoch += 1
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if self._early_stopping_patience > 0:
                if self._early_stopping_patience <= stop_iterations:
                    break
                # update stop iterations
                stop_iterations += 1
                if (last_test_loss - test_loss) > 1e-4:
                    stop_iterations = 0
                last_test_loss = test_loss
            elif epoch == epochs:
                break

        # store checkpoint when ending the training
        self.save_checkpoint(epoch)

        # return the losses for both training and validation
        return train_losses, test_losses

