import boto3
import os

bucket = 'aind-behavior-data'
s3_path_root = 'Han/ephys/'

local_cache_root = '/root/capsule/results/'


def export_df_and_upload(df, s3_rel_path, file_name):
    # save to local cache
    local_file_name = local_cache_root + file_name
    s3_file_name = s3_path_root + s3_rel_path + file_name

    df.to_pickle(local_file_name)
    size = os.path.getsize(local_file_name) / (1024 * 1024)

    # copy to s3
    res = upload_file(local_file_name, bucket, s3_file_name)
    if res:
        print(f'file exported to {s3_file_name}, size = {size} MB, df_length = {len(df)}')
    else:
        print('Export error!')
    return


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True