import csv
import datetime
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import copy

from P3INN import (
    DynamicLoss,
    PhysicsInformedNN,
    load_and_preprocess_data,
    plot_internal_states,
    plot_results,
    valid,
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"current GPU: {gpu_name}")

best_loss = float('inf')
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
print(current_time)
dataset_name = 'case'
sub_num = 30

for i in range(1, sub_num + 1):
    sub_id = i

    base_folder = f'{dataset_name}/sub{sub_id}'
    
    base_dir = os.path.join(base_folder, 'image')  
    os.makedirs(base_dir, exist_ok=True)
    
    phys_data_file = os.path.join(base_folder, 'physiological', 'physiological.csv')
    annotations_data_file = os.path.join(base_folder, 'annotations', 'annotations.csv')

    segment_length = 300
    random_seed = 2

    y_train, psi_train, y_val, psi_val, y_test, psi_test = load_and_preprocess_data(phys_data_file, annotations_data_file, random_seed=random_seed, segment_length = segment_length)

    mapping_functions = ['psi1', 'psi2', 'psi3', 'e', 'ln', 'tan']
    for mapping_func in mapping_functions:
        t = 0
        print(f"Training with mapping function: {mapping_func}")
        
        lr = 1e-3
        
        epochs = 1000
        batch_size_train = 64
        batch_size_test = 1
        train_data = TensorDataset(y_train, psi_train)
        val_data = TensorDataset(y_val, psi_val)
        test_data = TensorDataset(y_test, psi_test)

        train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size_test, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
        
        model = PhysicsInformedNN(input_size = segment_length*3 , output_size = segment_length ,mapping_function=mapping_func).to(device)
        loss_fn = DynamicLoss()  
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best_model_state = copy.deepcopy(model.state_dict())  
        best_logs = []
        best_test_mse = float('inf')
        best_test_mae = float('inf')
        best_train_mse = float('inf')
        best_epoch = -1
        logs = []
        log_header = ["epoch", "train_loss", "train_phy_loss", "train_psi_loss", "val_loss", "val_phy_loss", "val_MSE", "val_MAE", "val_R2"]

        u_pred = None

        for epoch in range(epochs+1):

            prev_model_state = copy.deepcopy(model.state_dict())
            prev_optimizer_state = copy.deepcopy(optimizer.state_dict())
            prev_logs = copy.deepcopy(logs)


            model.train()
            for y_batch, psi_batch in train_loader:
                y_batch = y_batch.to(device)
                psi_batch = psi_batch.to(device)
                psi_train_pred, u_net, u_eqn = model(y_batch)
                total_loss_train, phy_loss_train, psi_loss_train, train_mae = loss_fn(psi_train_pred, psi_batch, u_net, u_eqn)
                optimizer.zero_grad()
                total_loss_train.backward()
                optimizer.step()

                if torch.isnan(total_loss_train):
                    raise RuntimeError(f"NaN detected in training loss at epoch {epoch}")

            metrics_val = valid(model, val_loader, loss_fn, device)
            metrics_train = valid(model, train_loader, loss_fn, device)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} "
                    f"| Train phy loss: {metrics_train['phy']:.4f} | Train MSE: {metrics_train['mse']:.4f} | Train MAE: {metrics_train['mae']:.4f} "
                    f"| val MSE: {metrics_val['mse']:.4f} | val MAE: {metrics_val['mae']:.4f}"
                    f"| val phy loss: {metrics_val['phy']:.4f}")

                logs.append({
                    "epoch": epoch,
                    "train_loss": metrics_train['loss'],
                    "train_phy_loss": metrics_train['phy'],
                    "train_psi_loss": metrics_train['mse'],
                    "val_MSE": metrics_val['mse'],
                    "val_MAE": metrics_val['mae']
                })
        if epoch == epochs or t == 1:
            
            model.eval()
            

            
            with torch.no_grad():
                psi_train_pred, u_net_train, u_eqn_train = model(y_train)
                psi_test_pred, u_net_test, u_eqn_test = model(y_test)
                
                
            test_df = pd.DataFrame({
                "Sample_Index": np.arange(len(psi_test.cpu().numpy().flatten())),
                "True_Arousal": psi_test.cpu().numpy().flatten(),
                "Predicted_Arousal": psi_test_pred.cpu().numpy().flatten()
            })
            
            print("\nVisualization of the internal state of the training set:")
            plot_internal_states(y_train, u_net_train, u_eqn_train, sample_index=0)
            
            print("\nVisualization of the internal state of the test set:")
            plot_internal_states(y_test, u_net_test, u_eqn_test, sample_index=0)

            csv_test_path = os.path.join(base_dir, f'{current_time}_{mapping_func}_test_results.csv')
            test_df.to_csv(csv_test_path, index=False)
            
            plot_results(psi_train.cpu().numpy().flatten(),
                        psi_train_pred.cpu().numpy().flatten(),
                        save_dir=base_dir,
                        datetime_str=f'{current_time}',
                        dataset_type=f'train_{mapping_func}', 
                        loss=best_train_mse)
            
            plot_results(psi_test.cpu().numpy().flatten(),
                        psi_test_pred.cpu().numpy().flatten(),
                        save_dir=base_dir,
                        datetime_str=f'{current_time}',
                        dataset_type=f'test_{mapping_func}', 
                        loss=best_test_mse)
            
            mse = mean_squared_error(psi_test.cpu().numpy().flatten(), psi_test_pred.cpu().numpy().flatten())
            mae = mean_absolute_error(psi_test.cpu().numpy().flatten(), psi_test_pred.cpu().numpy().flatten())
            
            csv_path = os.path.join(base_dir, f'{current_time}_{mapping_func}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_header)
                writer.writeheader()
                writer.writerows(logs)
                param_writer = csv.writer(f)
                param_writer.writerow([])  
                param_writer.writerow(["Model Parameters"])
                param_writer.writerow(["Parameter", "testue"])
                param_writer.writerow(["mapping_function", mapping_func])
                param_writer.writerow(["test_mse", mse])
                param_writer.writerow(["test_mae", mae])
                param_writer.writerow(["a", model.mapping_params['a'].item()])
                raw_c = model.mapping_params['c']
                with torch.no_grad():
                    c_tensor = model.c_dynamic
                    c_mean = c_tensor.mean().item()
                param_writer.writerow(["c", c_mean])
                param_writer.writerow(["tau_r", model.tau_r().item()])
                param_writer.writerow(["tau_d", model.tau_d().item()])
                param_writer.writerow(["batch_size_train", batch_size_train])
                param_writer.writerow(["batch_size_test", batch_size_test])
                param_writer.writerow(["lr", lr])
            t=0