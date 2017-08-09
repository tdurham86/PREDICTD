'''This script should be run on a PREDICTD spark cluster master node just after 
the cluster is initialized and the user has run the 'aws configure' command to
supply their S3 credentials.
'''
import argparse
import os
import shutil
import subprocess
import sys

def set_up_spark_defaults(slave_type):
    sdefaults_path = '/root/spark/conf/spark-defaults.conf'
    sdefaults_temp = '/root/predictd/spark-defaults.conf.{!s}.temp'.format(slave_type)
    with open(sdefaults_path) as sdefaults_in:
        sdefaults_lines = []
        for line in sdefaults_in:
            if line[:15] in ['# for spark ver', 'spark.tachyonSt', 'spark.externalB']:
                sdefaults_lines.append(line)
    shutil.copy(sdefaults_temp, sdefaults_path)
    with open(sdefaults_path, 'a') as sdefaults_out:
        sdefaults_out.write(''.join(sdefaults_lines))

def set_up_spark_env():
    senv_path = '/root/spark/conf/spark-env.sh'
    senv_tmp = '/root/spark/conf/spark-env.sh.tmp'
    with open(senv_path) as senv_in, open(senv_tmp, 'w') as senv_out:
        for line in senv_in:
            if line.startswith('export SPARK_LOCAL_DIRS'):
                senv_out.write('export SPARK_LOCAL_DIRS="/data/spark"\n')
            else:
                senv_out.write(line)

def write_aws_creds(aws_key, aws_secret, def_region='us-west-2'):
    with open('/root/.aws/credentials', 'w') as out:
        out.write('[default]\n')
        out.write('aws_access_key_id = {!s}\n'.format(aws_key))
        out.write('aws_secret_access_key = {!s}\n'.format(aws_secret))

def subproc_call(cmd):
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        raise Exception('Error running the command: {!s}'.format(' '.join(cmd)))        

def copy_dirs_and_restart_spark():
    cmd = ['/root/spark/sbin/stop-all.sh']
    subproc_call(cmd)

    cmd = ['/root/spark-ec2/copy-dir', '/root/spark/conf']
    subproc_call(cmd)

    cmd = ['/root/spark-ec2/copy-dir', '/root/.aws']
    subproc_call(cmd)

    cmd = ['/root/spark/sbin/start-all.sh']
    subproc_call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aws_access_key')
    parser.add_argument('--aws_secret_access_key')
    parser.add_argument('--default_region', default='us-west-2')
    parser.add_argument('--slave_type', default='r3.8.xlarge')
    args = parser.parse_args()

    set_up_spark_defaults(args.slave_type)
    set_up_spark_env()
    write_aws_creds(args.aws_access_key, args.aws_secret_access_key, 
                    def_region=args.default_region)
    copy_dirs_and_restart_spark()
