
# Coupon Acceptance Prediction Pipeline

## Project Overview
This MLOps project aims to predict whether individuals will accept recommended coupons while in their vehicles. Utilizing the XGBoost model, we focus on creating an efficient MLOps pipeline to enhance prediction accuracy and streamline the deployment process.

## Data Source
We use the "In-Vehicle Coupon Recommendation" dataset from the UCI Machine Learning Repository. This dataset provides rich information about users, merchants, and coupon characteristics.

## Key Features
- Destination (No Urgent Place, Home, Work)
- Weather Conditions (Sunny, Rainy, Snowy)
- Time of Day (7AM, 10AM, 2PM, 6PM, 10PM)
- Coupon Categories (Restaurant(<$20), Restaurant($20-$50), Coffee House, Bar, Carry out & Take away)
- Expiration Time (2 hours, 1 day)
- Direction Alignment (0: No, 1: Yes)

## Technology Stack
- Python 3.9
- MLflow 2.4
- Prefect 2.11.2
- Docker
- AWS (EC2, S3, RDS)
- Terraform

## Core Components
1. **Infrastructure**: Terraform for AWS resource management
2. **Experimentation**: MLflow for experiment tracking and model registry
3. **Orchestration**: Prefect for workflow management and scheduling
4. **Deployment**: Docker for containerized model serving
5. **CI/CD**: GitHub Actions for automated pipeline execution

## Implementation Workflow
1. Set up AWS resources using Terraform
2. Conduct experiments and track results with MLflow
3. Orchestrate workflows using Prefect
4. Deploy models using Docker containers
5. Implement CI/CD pipeline with GitHub Actions

## Getting Started
1. Clone the repository
2. Set up the AWS infrastructure using Terraform
3. Install required Python packages
4. Run the main pipeline script

For detailed instructions, refer to the documentation in the `code/` directory.

## Contribution
We welcome contributions! Please see our contributing guidelines for more information.

## Acknowledgements
Special thanks to the UCI Machine Learning Repository for providing the dataset and to all contributors who have helped shape this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
