
1.First change the path to your code file and Install the required packages by using the following command.

D:\classification>pip install -r requirements.txt


2.please run the below command line and make sure to change your directory to your code path.

D:\classification>uvicorn classifier_app:app --reload


3.After running the code please run the below endpoints in your postman.

fetch and train endpoint:
-------------------------
the below url is used to fetching the customer data from the postegresql server (or) customer_data, preprpcessing, training and saving the best model.
method:post
url: http://127.0.0.1:8000/auto_train_assigned_group/

payloads/parameters:

{
"customer": "customer"
}

results endpoint:
-----------------
the below url is used to loading the model  and returning the results in json format.

method:post
url: http://127.0.0.1:8000/predict_assigned_group/

payload:

{
    "customer":"customer"
}
