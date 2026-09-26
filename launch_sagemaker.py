import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role

# 1. Setup SageMaker session and role
# If running this locally, you'll need to configure AWS CLI and specify the role explicitly
# role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
try:
    role = get_execution_role()
except ValueError:
    print("Execution role not found. Please provide it manually if running locally.")
    role = "YOUR_SAGEMAKER_EXECUTION_ROLE_ARN_HERE"

sagemaker_session = sagemaker.Session()

# 2. Specify the S3 path where your data is located
# Update this if your bucket has a different name
s3_data_path = 's3://your-bucket-name/student_resource/'

# 3. Create the Estimator
# We use the SKLearn estimator which easily handles Python scripts and requirements.txt

# Specify your GitHub repository here
git_config = {
    'repo': 'https://github.com/your-username/your-repo-name.git',
    'branch': 'main'
}

estimator = SKLearn(
    entry_point='src/05_train_matcher.py', # The path inside your GitHub repo
    source_dir='.',                        # The root of the git repo
    git_config=git_config,                 # Tell SageMaker to clone this repo
    dependencies=['requirements.txt'],     # SageMaker will pip install these automatically
    role=role,
    instance_count=1,
    instance_type='ml.m5.4xlarge',         # CPU instance with good RAM. Change to ml.g4dn.xlarge for GPU
    framework_version='1.2-1',             # Scikit-learn version
    py_version='py3',
    sagemaker_session=sagemaker_session,
    base_job_name='entity-matcher-training'
)

# 4. Start the training job
print(f"Starting training job. Data will be pulled from: {s3_data_path}")
estimator.fit({'train': s3_data_path})

print("\nTraining Job Complete!")
print("You can download the resulting model (entity_matcher_catboost.cbm) from the S3 model artifact URI:")
print(estimator.model_data)
