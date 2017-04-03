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
    1. Go to the ```/root/spark/conf``` directory and edit the ```spark-defaults.conf``` file to include the following lines:
        ```
        #The number of executors to run on the cluster (usually set to the 
        #number of available worker node cores divided by the cores per executor,
        #specified in the spark.executor.cores parameter below.
        spark.executor.instances 16
        
        #The number of cores that should be accessible for each executor. A single executor
        #can run multiple jobs in parallel, but must exist on a single machine. I have found
        #that using 4 cores per executor works well.
        spark.executor.cores 4
        
        #The amount of memory available to each executor. Usually works to set this to the
        #total available worker memory divided by the number of executors, with two GB or so
        #subtracted to allow some overhead.
        spark.executor.memory 54g
        
        #The number of cores to use on the driver program, which runs on the master node.
        spark.driver.cores 2
        
        #The amount of memory the driver program can use on the master node.
        spark.driver.memory 12g
        
        #Related to the amount of memory available to the driver; I have found it
        #helpful to keep this value pretty close to the spark.driver.memory value.
        spark.driver.maxResultSize 10g
        ```
        The ```spark-defaults.conf``` file contains parameters describing the allocation of resources and the behavior of the Spark distributed processing engine. The values shown here assume that the cluster contains a ```m4.xlarge``` master node and a single ```x1.16xlarge``` worker node. If you run different instance types or different numbers of instances you may have to adjust these values to fit the new cluster resources.
    1. Edit the first line of the ```spark-env.sh``` file that sets the ```SPARK_LOCAL_DIRS``` environment variable to read:
        ```
        export SPARK_LOCAL_DIRS="/data/spark"
        ```
        This is the directory where Spark will write temporary files associated with the jobs that are running. The PREDICTD AMI comes with a 500 GB EBS drive that is mounted under ```/data```, and we recommend that you specify a subdirectory on this drive.
    1. Last, open the ```log4j.properties.template``` file, and edit the first uncommented line to read:
        ```
        log4j.rootCategory=WARN, console
        ```
        Then, save the file as ```log4j.properties```. This modification limits the number of messages that get logged to the console and turns on a progress bar that is useful for tracking the progress of jobs that run for more than a second or two.

    1. Set your S3 access key information as environment variables in the same way that you did when setting up the cluster.
        ```bash
        export AWS_ACCESS_KEY_ID=<AWS_ACCESS>
        export AWS_SECRET_ACCESS_KEY=<AWS_SECRET> 
        ```
    1. The PREDICTD code is stored in the ```/root/predictd``` directory. Go there and create a file called ```s3_credentials.txt``` to store your S3 access credentials for PREDICTD to read in. The file contents should look something like this:
        ```
        aws_access_key_id=<AWS_ACCESS_KEY_ID_HERE>
        aws_secret_access_key=<AWS_SECRET_ACCESS_KEY_HERE>
        host=<S3_REGION_HOST_HERE>
        ```
        The region host is a string like ```s3-us-west-2.amazonaws.com```, and will depend on the region in which you set up your S3 account.
        
    1. Make sure you have an S3 bucket in which to store the PREDICTD output. You can easily create buckets using the browser-based S3 console.
    1. Navigate to the ```/root/predictd``` directory and edit the ```run_demo.sh``` script to point to the name of the bucket to which you would like the output written, as well as the root of the output S3 keys that you would like to use.
    1. You are now ready to run the PREDICTD demo. Simply navigate to the ```/root/predictd``` directory and run the following command:
        ```bash
        spark-submit ./run_demo.sh
        ```
        This script will call the ```impute_roadmap_consolidated.py``` PREDICTD script, which will access the Epigenomics Roadmap data in a publicly-accessible S3 bucket and train a model based on the first test split and first validation split from our published experiments. After training the model on the ENCODE Pilot Regions, it will save the model parameters and the imputed and observed data tracks to S3 in the bucket and root key that you specified in the ```run_demo.sh``` script.
    1. View the generated tracks in the UCSC Genome Browser by downloading the ```bigwigs/track_lines.txt``` file from the results stored in S3, and then copy those lines into the "Paste URLs or data" box in the "Manage Custom Tracks" -> "Add custom tracks" page on the human genome browser.
