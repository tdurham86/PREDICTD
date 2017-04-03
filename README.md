# PREDICTD
Durham T, Libbrecht M, Howbert J, Bilmes J, Noble W. PaRallel Epigenomics Data Imputation with Cloud-based Tensor Decomposition. 2017.

This repository contains the code to run PREDICTD, a program to model the epigenome based on the Encyclopedia of DNA Elements and the NIH Roadmap Epigenomics Project data and to impute the results of epigenomics experiments that have not yet been done. A computing environment for running this code is distributed as an Amazon Machine Image, and it is easiest to get the code up and running by following the steps in the tutorial below to start a cluster in Amazon Web Services.

## Demo Tutorial

#### Set up Amazon Web Services accounts

1. Ensure that you have an account for Amazon Web Services. Set up a Simple Storage Service (S3) bucket and record the region, access key, and secret access key for a bit later.

#### Start a cluster with ```spark-ec2```

1. Use the AWS Management Console to make a small AWS instance to use for starting the cluster. Create a small Elastic Compute Cloud (EC2) instance like m3.medium (using a spot instance will make it cheaper). Search for and select the PREDICTD Amazon Machine Image (AMI) when you have the option.

1. Once the instance is ready, ssh to it, and set your S3 credentials as environment variables.

    ```bash
    export AWS_ACCESS_KEY_ID=<AWS_ACCESS>
    export AWS_SECRET_ACCESS_KEY=<AWS_SECRET> 
    ```
    You should also generate an EC2 key pair and put it on the new EC2 instance. See AWS instructions [here](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html).

1. Now, start up the cluster with a command line like the following. We recommend using a ```m4.xlarge``` instance for the head node, and a ```x1.16xlarge``` instance for the worker node. Always use spot instances for the workers, as this can save up to 90% on the cost of the workers.

    ```bash
    /home/ec2-user/spark-1.6.0/ec2/spark-ec2 --key-pair=EC2_keypair_name --identity-file=/path/to/EC2_keypair.pem --region=us-west-2 --zone=us-west-2a --ami=<PREDICTD_AMI> --master-instance-type=m4.xlarge --instance-type=x1.16xlarge --spot-price=2.00 --slaves=1 --spark-version=1.6.0 --hadoop-major-version=yarn --copy-aws-credentials --instance-profile-name=EMR_EC2_DefaultRole --ganglia launch predictd-demo
    ```
    This will create two new instances in your AWS management console, one called predictd-demo-master and the other called predictd-demo-slave.

1. SSH to the predictd-demo-master machine to configure Spark and run the demo.
