import uvicorn
import os
import json
import pathlib
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import classifier as da


class Item(BaseModel):
    customer: str
    summary:  Optional[str] = "None"
    col_name_summary: Optional[str] = "None"
    col_name_assigned: Optional[str] = "None"

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = pathlib.Path(__file__).parent.absolute()
os.chdir(base_dir)
base_cwd = os.getcwd()

@app.post("/auto_train_assigned_group/")
async def train_assignee(item: Item):
        cust = item.customer
        sum_col = item.col_name_summary
        assigned_col = item.col_name_assigned
        #output = da.initiate_process(cust, sum_col, priority_col)
        try:
            output = da.initiate_process(cust, sum_col, assigned_col)
            headers = {"Access-Control-Allow-Origin": "*"}
            return JSONResponse(content=output, headers=headers)
        except:
            output = {"Error": "error in training or columns are not present"}
        return output

@app.post("/predict_assigned_group/")
async def get_assignee(item: Item):
        cust = item.customer
        summ = item.summary
        #output = pc.detect_assigned_group(cust, summ)
        try:
            output = da.detect_assigned_group(cust, summ)
            headers = {"Access-Control-Allow-Origin": "*"}
            return JSONResponse(content=output, headers=headers)
        except:
            output = {"Error": "error in loading model"}
        return output
#####################################
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)

