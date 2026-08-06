import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy import signal
from scipy.interpolate import interp1d


def preprocess_phys_data(phys_data_file, target_sampling_rate=25):
    
    phys_df = pd.read_csv(phys_data_file)

    
    phys_df = phys_df[['gsr', 'video']]

    
    original_sampling_rate = 1000
    downsample_factor = original_sampling_rate // target_sampling_rate  

    
    gsr_resampled = signal.decimate(phys_df['gsr'].values, downsample_factor, ftype='iir')

    
    video_resampled = phys_df['video'].values[::downsample_factor]

    
    min_len = min(len(gsr_resampled), len(video_resampled))
    
    phys_df_down = pd.DataFrame({
        'gsr': gsr_resampled[:min_len],
        'video': video_resampled[:min_len]
    })
    
    
    phys_df_down.reset_index(drop=True, inplace=True)

    
    eda_clean = nk.eda_clean(phys_df_down['gsr'].values, sampling_rate=target_sampling_rate, method='neurokit')

    
    eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=target_sampling_rate, method='highpass')

    
    phys_df_down['gsr_clean'] = eda_clean
    phys_df_down['scr'] = eda_decomposed["EDA_Phasic"].values

    return phys_df_down


def upsample_annotations(annotations_df, original_sr=20, target_sr=25):
    if len(annotations_df) == 0:
        return annotations_df
        
    n_samples_original = len(annotations_df)
    
    duration = (n_samples_original - 1) / original_sr
    
    
    t_original = np.linspace(0, duration, n_samples_original)
    t_new = np.arange(0, duration + 1e-9, 1 / target_sr) 
    
    df_upsampled = pd.DataFrame(index=range(len(t_new)))
    
    for col in annotations_df.columns:
        
        is_discrete = (col == 'video') or (not pd.api.types.is_numeric_dtype(annotations_df[col]))
        
        if is_discrete:
            
            cat_obj = pd.Categorical(annotations_df[col])
            codes = cat_obj.codes
            
            
            if np.any(codes == -1):
                codes = pd.Series(codes).ffill().bfill().values.astype(int)

            f = interp1d(t_original, codes, kind='nearest', fill_value='extrapolate')
            new_codes = np.round(f(t_new)).astype(int)
            
            
            reconstructed_values = cat_obj.categories[new_codes]
            df_upsampled[col] = reconstructed_values
            
            
            if pd.api.types.is_integer_dtype(annotations_df[col]):
                df_upsampled[col] = df_upsampled[col].astype(int)
                
        else:
            
            f = interp1d(t_original, annotations_df[col], kind='linear', fill_value='extrapolate')
            df_upsampled[col] = f(t_new)
            
    
    df_upsampled.reset_index(drop=True, inplace=True)
    return df_upsampled


def filter_video_data(phys_df, annotations_df):
    
    
    mask = ~phys_df['video'].isin([10, 11, 12])
    
    
    phys_df_filtered = phys_df[mask].reset_index(drop=True)
    annotations_df_filtered = annotations_df[mask].reset_index(drop=True)

    return phys_df_filtered, annotations_df_filtered


def save_data(phys_df, annotations_df, subject_id):

    output_dir = os.path.join('case', f'sub{subject_id}')
    physiological_dir = os.path.join(output_dir, 'physiological')
    annotations_dir = os.path.join(output_dir, 'annotations')

    os.makedirs(physiological_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    
    phys_df.to_csv(os.path.join(physiological_dir, 'physiological.csv'), index=False)

    
    annotations_df.to_csv(os.path.join(annotations_dir, 'annotations.csv'), index=False)

    print(f"Data saved for subject {subject_id}.")


def process_subject_data(subject_id, phys_data_file, annotations_data_file):
    
    annotations_df = pd.read_csv(annotations_data_file)
    annotations_df = upsample_annotations(annotations_df, original_sr=20, target_sr=25)

    
    phys_df_down = preprocess_phys_data(phys_data_file)

    
    phys_df_filtered, annotations_df_filtered = filter_video_data(phys_df_down, annotations_df)

    
    save_data(phys_df_filtered, annotations_df_filtered, subject_id)

if __name__ == '__main__':
    sub_num = 30
    dataset_name = 'case'

    for subject_id in range(1, sub_num + 1):
        
        annotations_data_file = os.path.join('data', dataset_name, 'interpolated', 'annotations', f'sub_{subject_id}.csv')
        phys_data_file = os.path.join('data', dataset_name, 'interpolated', 'physiological', f'sub_{subject_id}.csv')
    
        process_subject_data(subject_id, phys_data_file, annotations_data_file)