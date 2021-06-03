The notebook [Case_Study.ipynb](Case_Study.ipynb) contains all details on how to run SageMaker training jobs with profiling enabled. It showcases different training configurations such:
- single GPU training (![train.py](entry_point/train.py))
- mulit GPU training using PyTorch DataParallel (![train.py](entry_point/train.py))
- multi GPU training using PyTorch DistributedDataParallel ![train_ddp.py](entry_point/train_ddp.py)


The notebook also shows how to create the visualizations of profiling data while the training is in progress. You can find an executed version of the notebook in [Case_Study.html](Case_Study.html) 
that includes all interactive visualizations and graphs and in [Case_Study.pdf](Case_Study.pdf)

Debugger's profiling feature generates automatic reports for each training job. We have uploaded the reports for each training configuraton that were run in the notebook.
- single GPU training: ![profiler-report-1.html](profiler-report-1.html)
- mulit GPU training using PyTorch DataParallel:  ![profiler-report-2.html](profiler-report-2.html)
- multi GPU training using PyTorch DistributedDataParallel: ![profiler-report-3.html](profiler-report-3.html)
- sngle GPU optimized training: ![profiler-report-4.html](profiler-report-4.html)
