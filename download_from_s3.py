import boto3
import os

aws_access_key_id = 'AKIAYAV34GOXJ2BZLK66'
aws_secret_access_key = '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'
region_name = 'ap-south-1'
bucket_name = 'pose-estimation-internship'

def download_file_from_s3(file_name, download_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    try:
        s3.download_file(bucket_name, file_name, download_path)
        print(f'{file_name} downloaded successfully to {download_path}.')
    except Exception as e:
        print(f'Error downloading {file_name} from {bucket_name}: {e}')

#downloading a file
file_name = 'processed_data.csv'  # The name of the file in the S3 bucket
download_path = r"D:\project\infosys_project\csv_data\processed_data.csv"  # Local path to save the downloaded file

download_file_from_s3(file_name, download_path)
