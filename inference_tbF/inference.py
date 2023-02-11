#!/usr/bin/env python
import onnxruntime as rt
import pandas as pd
import numpy as np
import soundfile as sf


def prediction(input_path, model_path):	
    switch = 1
    AllCol = ['participant', 'filename', 'age', 'height', 'weight',
          'reported_cough_dur', 'heart_rate', 'temperature', 'sex_Female',
          'sex_Male', 'tb_prior_No', 'tb_prior_Not sure', 'tb_prior_Yes',
          'tb_prior_Pul_No', 'tb_prior_Pul_Yes', 'tb_prior_Extrapul_No',
          'tb_prior_Extrapul_Yes', 'tb_prior_Unknown_No',
          'tb_prior_Unknown_Yes', 'hemoptysis_No', 'hemoptysis_Yes',
          'weight_loss_No', 'weight_loss_Yes', 'smoke_lweek_No',
          'smoke_lweek_Yes', 'fever_No', 'fever_Yes', 'night_sweats_No',
          'night_sweats_Yes']
    CODAMuStd = np.asarray([[ 40.817745 , 161.45787  ,  57.327652 ,  46.227486 ,  87.421104 ,
            36.733143 ],
          [ 15.1564245,   8.7601595,  13.59417  ,  51.92165  ,  16.980263 ,
              0.5627685]])
    input_df = pd.read_csv(input_path+'meta_info.csv')[['participant', 'filename']]
    X_list_label = np.zeros((input_df.shape[0], 27))

    try:
        input_df2 = pd.read_csv(input_path+'CODA_TB_Clinical_Meta_Info_Test.csv')
        input_df3 = pd.merge(input_df2, input_df, on="participant")
        numerical = input_df3[['filename', 'participant', 'age', 'height', 'weight', 'reported_cough_dur', 'heart_rate',
            'temperature']].copy()
        categorical = input_df3[['sex', 'tb_prior', 'tb_prior_Pul',
            'tb_prior_Extrapul', 'tb_prior_Unknown', 'hemoptysis', 'weight_loss', 'smoke_lweek', 'fever', 'night_sweats', 'participant']].copy()
        categorical = pd.get_dummies(categorical).copy()
        categorical['filename'] = input_df3['filename']
        input_df3 = pd.merge(numerical, categorical, on="filename")
        temp = input_df3[['participant', 'filename']].copy()
        input_df = temp
        
        switch = 0        
        for i in range(6):
            try:
                input_df3[AllCol[2:][i]] = (input_df3[AllCol[2:][i]].astype('float32')-CODAMuStd[0, i])/CODAMuStd[1, i]
                input_df3[AllCol[2:][i]] = input_df3[AllCol[2:][i]].fillna(0).copy()
                temp = input_df3[AllCol[2:][i]].values
                X_list_label[:, i] += temp
            except:
                X_list_label[:, i] = np.zeros(X_list_label.shape[0])
                
        for i in range(21):
            try:
                temp = input_df3[AllCol[2:][i+6]].values
                X_list_label[:, i+6] += temp
            except:
                X_list_label[:, i+6] = np.zeros(X_list_label.shape[0])
    except:
        input_df = pd.read_csv(input_path+'meta_info.csv')[['participant', 'filename']]

    dataXList = input_df['filename'].values
    X_list = []
    for i in range(len(dataXList)):
        patient = np.zeros((22050,1), dtype=np.float32)
        rawValues, rate = sf.read(input_path+'raw_test_data/'+dataXList[i])
        mu = np.nanmean(rawValues)
        std = np.nanstd(rawValues)
        patient[-len(rawValues):, 0] = (rawValues-mu)/std
        X_list.append(patient)
    X_list = np.asarray(X_list)

    pred = np.zeros(X_list.shape[0])
    model_inference = rt.InferenceSession(model_path+'model.onnx')
    input_name = model_inference.get_inputs()[0].name
    input_name2 = model_inference.get_inputs()[1].name
    label_name = model_inference.get_outputs()[0].name
    label_name2 = model_inference.get_outputs()[1].name    
    for i in range(X_list.shape[0]):
        onnx_pred = model_inference.run([label_name, label_name2], {input_name: X_list[i:i+1, :].astype(np.float32), input_name2: X_list_label[i:i+1, :].astype(np.float32)})
        pred[i] += onnx_pred[switch][0, 0]

    #Paitent id
    df_pred = pd.DataFrame(input_df['participant'], columns=['participant']) 
    df_pred['probability'] = pred
    df_pred_temp = df_pred.dropna().groupby('participant')['probability'].max()
    df_pred_temp_values = df_pred_temp.values
    df_pred_temp_values[df_pred_temp_values>=1] = 1
    df_pred_temp_values[df_pred_temp_values<0] = 0
    df_pred_fin = pd.DataFrame(np.asarray([df_pred_temp.index, df_pred_temp_values]).T, columns=['participant','probability'])
    df_pred_fin.to_csv('./output/predictions.csv', index=False)

    # comfirm that probability of predictions.csv exists
    print(pd.read_csv('./output/predictions.csv')['probability'])
    print('mode :', switch)


if __name__ == '__main__':    
    #holdout sample file path 
    input_path = './input/' # Utilize {'meta_info.csv'  'raw_test_data/*.wav'} according to your preprocessing logic
    # In SC2 you may also load CODA_TB_Clinical_Meta_Info_Test.csv
    # however attempting to load it in SC1 will result in failure
    
    #model path
    model_path = './model/'
    prediction(input_path, model_path)
