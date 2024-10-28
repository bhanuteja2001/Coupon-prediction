
```markdown


## Setup and Installation

1. Install pipenv:
   ```
   pip install pipenv
   ```

2. Install project dependencies:
   ```
   pipenv install pandas scikit-learn mlflow prefect --python==3.9
   ```

3. Install development dependencies:
   ```
   pipenv install --dev pylint black isort pre-commit pytest
   ```

4. Set up pre-commit hooks:
   ```
   pre-commit install
   ```

## Configuration

1. Adjust settings in `pyproject.toml` and `.pre-commit-config.yaml` for formatting and linting.

2. Set up MLflow tracking server:
   - Use Terraform to create AWS resources (EC2, S3, RDS). See [Terraform setup](https://github.com/bhanuteja2001/coupon-prediction/tree/code/infrastructure).
   - Configure services following [these instructions](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md).

## Workflow Orchestration with Prefect

1. Start Prefect server:
   ```
   prefect server start
   ```
   or
   ```
   prefect cloud login
   ```

2. Initialize Prefect project:
   ```
   prefect project init
   ```

3. Create a work pool:
   ```
   prefect work-pool create my-pool
   ```

4. Start a worker:
   ```
   prefect worker start -p my-pool -t process
   ```

5. Create a deployment for hyperparameter tuning:
   ```
   prefect deploy code/model.py:flow -n 'deployment' -p my-pool
   ```

6. Track experiments using MLflow UI at `http://aws_bucket_uri:5000/`

## Model Registration

After experiments, register the best model:
```
python best_model.py
```
Alternatively, create a Prefect deployment for this step.

## Prediction Server

1. Build Docker image:
   ```
   docker build -t coupon-accepting-prediction-service:v1 .
   ```

2. Run the server:
   ```
   docker run -it --rm -p 9696:9696 coupon-accepting-prediction-service:v1
   ```

3. To pass AWS credentials for loading the model from S3:
   ```
   docker run -it --rm -p 9696:9696 \
     -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
     -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
     coupon-accepting-prediction-service:v1
   ```

   Set AWS credentials:
   ```
   AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)
   AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)
   ```

## Usage

1. Train the model and tune hyperparameters using Prefect deployments.
2. Register the best model using `best_model.py`.
3. Deploy the prediction service using Docker.
4. Send prediction requests to the deployed service.


