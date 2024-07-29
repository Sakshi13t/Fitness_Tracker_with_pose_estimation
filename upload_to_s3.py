import boto3
import os

aws_access_key_id = 'key-id'
aws_secret_access_key = 'secret-key'
region_name = 'region'
bucket_name = 'bucket-name'

# Establishing a connection to AWS S3
s3 = boto3.resource(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

def upload_file_to_s3(file_path):
    file_name = os.path.basename(file_path)
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    try:
        response = s3.upload_file(file_path, bucket_name, file_name)
        print(f'{file_name} uploaded successfully to {bucket_name}.')
    except Exception as e:
        print(f'Error uploading {file_name} to {bucket_name}: {e}')

new_file_path = "D:\project\infosys_project\csv_data\processed_data.csv"
upload_file_to_s3(new_file_path)
