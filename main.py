from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import io
import joblib

app = FastAPI(title=".csv -> Model -> .csv", version = '1.0')

model_dict = {'CaO':joblib.load('../models_scalers/ca_model.joblib'),
              'SiO2':joblib.load('../models_scalers/si_model.joblib'),
              'Al2O3':joblib.load('../models_scalers/al_model.joblib'),
              'Fe2O3':joblib.load('../models_scalers/fe_model.joblib')
             }

scaler_dict = {'CaO':joblib.load('../models_scalers/ca_scaler.joblib'),
               'SiO2':joblib.load('../models_scalers/si_scaler.joblib'),
               'Al2O3':joblib.load('../models_scalers/al_scaler.joblib'),
               'Fe2O3':joblib.load('../models_scalers/fe_scaler.joblib')
              }

requirements = {
    'CaO':['b3','Fe_Ca','Al_Si_kmeans'],
    'SiO2':['b5','Fe_Ca','Al_Si_kmeans'],
    'Al2O3':['b5','Fe_Ca','Al_Si_kmeans'],
    'Fe2O3':['b5','Fe_Ca','Al_Si_kmeans'],
                }

def predict_with_each_model(model, scaler, dependency_columns, sample):
    try:
        # Fix accidental double brackets
        if len(dependency_columns) == 1 and isinstance(dependency_columns[0], list):
            dependency_columns = dependency_columns[0]

        scaled_sample = scaler.transform(sample[dependency_columns]) # Scaling sample
        y_pred = model.predict(scaled_sample) # Predicting true oxide values for sample
        return y_pred
        #return round(float(y_pred),2)  # Convert numpy float to Python float
    except Exception as e:
        print('fail 4')
        print(f"Error in prediction: {e}")
        


@app.post('/predict-csv')
async def predict_csv(
    file: UploadFile = File(...),
    include_inputs: bool = Query(default=False, description='Return both inputs and predictions if true')
):
    
    # 1 Basic content-type check

    # 2 Read csv into DataFrame
    try:
        df = pd.read_csv(file.file, sep=';') # Gets file from file.file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {e}")
    
    # 3 Validate shape and columns
    if df.shape[1] != 4:
        raise HTTPException(
                status_code=422, detail=f"Expected 3 columns, got {df.shape[1]} columns: {list(df.columns)}"
                )
       
    # 4 Predict
    try:
        out_df = pd.DataFrame()
        for key,model in model_dict.items():
            # .ravel() removes a dimension in the retuned np.array([[]]), to just be np.array([])
            out_df[key] = predict_with_each_model(model, scaler_dict[key],[requirements[key]], df).ravel().round(3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
        
    # 5 Build response CSV
    buffer = io.StringIO() # Start a text-stream to memory
    out_df.to_csv(buffer,index=False) # Writes df to memory as text stream
    buffer.seek(0) # Moves pointer back to beginning, so streaming response will start from beginning
    
 
    return StreamingResponse(
        buffer, # Streamingresponse gets the tet stream in memory
        #iter([buffer.getvalue()]),
        media_type="text/csv", # States this is a csv file
        headers={"Content-Disposition": "attachment; filename=predictions.csv"} # Forces client to download file with filename
    )
    
