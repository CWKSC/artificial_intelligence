import math
from pathlib import Path
from typing import Callable, Literal, Union
import numpy
import torch
from tqdm import tqdm

current_file_directory: Path = None
device: str = None

def init(file_path: str):
    global current_file_directory
    current_file_directory = Path(file_path).parent

    global device
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using {device} device')


def train(
    model: torch.nn.Module,
    target_tensors: torch.Tensor, 
    input_tensors: torch.Tensor, 
    repeat: int = 10000,
    loss_fn = torch.nn.BCEWithLogitsLoss(), 
    optimizer = None,
    correct_func: Callable[[torch.Tensor, torch.Tensor], bool] = lambda pred, target: pred == target,
    save_mode: Union[Literal['accuracy'], Literal['loss'], None] = "accuracy",
    save_dir_path: str = 'model',
    save_file_name: str = 'NN'
) -> None:
    try:
        if save_mode != None:
            directory = current_file_directory / save_dir_path
            directory.mkdir(parents=True, exist_ok=True)
        
        if optimizer == None:
            optimizer = torch.optim.RAdam(model.parameters())
            
        model.to(device)
        model.train()

        best_loss = float('inf')
        best_accuracy = 0
        tensor_len = len(target_tensors)

        for epoch in range(1, repeat + 1):
            print(f'Epoch {epoch}')

            total_loss = 0
            total_correct = 0

            for target, input in tqdm(list(zip(target_tensors, input_tensors))):

                target = target.to(device)
                input = input.to(device)

                pred = model(input)
                loss = loss_fn(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not numpy.isnan(loss.item()):
                    total_loss += loss.item()
                total_correct += 1 if correct_func(pred, target) else 0
            
            if total_loss < best_loss:
                best_loss = total_loss
                if save_mode == 'accuracy':
                    torch.save(model.state_dict(), directory / f'{save_file_name}.pt')
            
            accuracy = total_correct / tensor_len
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if save_mode == 'loss':
                    torch.save(model.state_dict(), directory / f'{save_file_name}.pt')

            print(f"loss:     {total_loss:>7f}, best_loss: {best_loss:>7f}")
            print(f"accuracy: {accuracy:>7f}, {total_correct} / {tensor_len}, best_accuracy: {best_accuracy:>7f}")
    except KeyboardInterrupt:
        pass

def eval(
    model: torch.nn.Module,
    target_tensors: torch.Tensor, 
    input_tensors: torch.Tensor, 
    loss_fn = torch.nn.BCEWithLogitsLoss(), 
    optimizer = None,
    correct_func: Callable[[torch.Tensor, torch.Tensor], bool] = lambda pred, target: pred == target
):
    if optimizer == None:
        optimizer = torch.optim.RAdam(model.parameters())
    model.to(device)
    model.eval()

    tensor_len = len(target_tensors)
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for target, input in tqdm(list(zip(target_tensors, input_tensors))):

            target = target.to(device)
            input = input.to(device)
            
            predict = model(input)
            loss = loss_fn(predict, target)

            if not numpy.isnan(loss.item()):
                total_loss += loss.item()
            total_correct += 1 if correct_func(predict, target) else 0
        
    print(f"loss: {total_loss:>7f} ")
    print(f"accuracy: {total_correct / tensor_len}, {total_correct} / {tensor_len}")

def predict(
    model: torch.nn.Module,
    input_tensors: torch.Tensor, 
) -> list[torch.Tensor]:
    model.to(device)
    model.eval()
    predict_list = []
    with torch.no_grad():
        for input in tqdm(input_tensors):
            input = input.to(device)
            predict = model(input)
            predict_list.append(predict)
    return predict_list

def load_model(model: torch.nn.Module, file_path: str) -> torch.nn.Module:
    state_dict = torch.load(current_file_directory / (file_path + '.pt'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compare_float_true_false(pred, target):
    pred = pred.item()
    target = target.item()
    pred = 1 if pred > 0.5 else 0
    return pred == target

def compare_float_isclose(pred, target):
    pred = pred.item()
    target = target.item()
    return numpy.isclose(pred, target)
