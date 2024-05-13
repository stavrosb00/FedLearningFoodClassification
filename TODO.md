# TODO

## Functional
- [✓] dirichlet uni split -> stratified split | an min_samples>2 tote dinw se val split part alliws ta krataei ola to train split part
- [✓] koinh bash gia ola ta peiramata me fixed template | bs=32, val_rate=0.3
- [✓] strategy saves each client’s metrics (fit,eval stats etc.) on .csv for accountability
- [✓] Scaffold fix | aggr Dc, aggr Buffers, aggr Dx
- [✓] Strategy saves checkpoints based on test accuracy 
https://flower.ai/docs/framework/how-to-save-and-load-model-checkpoints.html
- [✓] Best global test acc calculated after evaluate_fn -> Save as params, loss, acc, round .npz format. Load them on main.py. [possible] h5py hdf5 arrays
- [semi] Save server_cv on Server class side for warm start expirement => [NotNeeded]. Apla gia inference na krataw architechture params 
- [✓] BEST aggregated validation acc after eval_round
- [✓] Bash commands as commented on .gitignore for mannually MB cleanups on .pt, .npz and .pth
- [✓] Average fit mins h secs se enan client basei tou telikou plot sto sampled .csv
- [✓] visual.ipynb: Plot client progress based on ticks and interp between ticks. [possible] Abuse config.yaml or plot in every end of expirement on main.py internally
- [✓] Na balw kai autocast sthn aplh train(): Time sygkrish 20 mins grhgorotera se 200 rounds. Apodosh sygkrish(6/4 stis 11:33 VS stis 21:12): Kalytera stis 21:12
- [] Retrieve history from results of exp to add test acc outputs\2024-04-05\21-50-30 #239 mins runtime 
- [✓] outputs\2024-04-06\11-33-29 alpha=0.1 kai idia settings me to 18    : VS outputs\2024-04-06\21-12-08 a=0.1 XWRIS AMP
- [✓] Compare partitioning gia IID,Dir0.5, Dir0.1 : fedavg, fedprox, scaffold.. exp_files ktlp.
- [✓] Scaffold me 2x state_dicts opws Xtra-Computing kai compare me already scaffold 150-200 rounds : https://github.com/Xtra-Computing/NIID-Bench/blob/main/experiments.py#L297
- [✓] NEED TO DROP TRAIN AMP floating points na dw drop precision kai anebasma batch size isws (torch.amp for autocast and grad scaler, lighting , quantization). Na dw isws 
gia resize kai downsample foodset -> epirroh se training acc k timers. 
https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/ Best practices section
https://pytorch.org/docs/stable/amp.html
https://discuss.pytorch.org/t/nan-loss-with-torch-cuda-amp-and-crossentropyloss/108554/18
https://pytorch.org/mobile/home/
= [✓] Time and GB comparison w/ & w/out torch.amp| saves 400 MB for Server, 1.7 GB for Client process, 7sec/round 
- [] Shell script for Clients = 10,16 ,NumClasses = 4, 10, (101),
- [✓] Table me 3 algorithms kai best_val_accuracies . Bold to best twn triwn
- [] Dirichlet 0.1 expirements gia fedavg, fedprox, OK-scaffold
- [✓] Fixed summary dirichlet plot
- [✓] LaTeX table me accs kai figure plots
- [✓] LaTeX plot client progress plots
- [✓] Data-vis plots sto overleaf
- [✓] Pre-process samples me 2 eikones sto idio grid: mia prin kai mia meta. + Intensity histogram pou fainetai to uint8 k to 0...1 vs 0...255
- [✓] LaTeX pre-process figures
- [✓] Latex list of symbols(comm rounds T, N clients, Si sub-synolo k.a.)
- [✓] Dirichlet 0.1 SCAFFOLD full sample fraction
- [✓] Latex more plots KAI MORE SXOLIA 
- [semi] Latex algorithms  
- [] Parse model architecture through arguments. Config .yaml _target + python instances
- [✓] Two crops and data augmentation , memory_loader, 
- [✓] Functioning MLP projector & predictor , SimSiam arch
- [✓] Optimizer SGD with init_lr and learning scheduler cosine decay
- [✓] Na dw bibliografia Fed gia memory vs test dataloader se knn monitoring. To FedSSL divergence paper exei training dataloader se server side. Alliws na kanw weighted kNN accs
- [✓] Na dw alignment dataset me 0 validation support
- [✓] Alignment dataloader optimization kai profiling benchmark gia tensor or numpy choose gia LinearCKA
- [semi] Client, strategy kai server gia HeteroSSFL
- [semi] tensorboard real time visuals
- [] main_ssfl script gia metrics saves kai instances,calls.
- [] Linear evaluate to megalo peirama 12 Maiou
- [] Delete 80 test peiramata days : 13/5 , 
- [] Load kai kNN accuracy sto main_ssfl
- [] Linear evaluation : analoga .npz h .pth na kanei manipulation to load twn checkpoints
- [] Linear evaluation : load_downstream_data() synarthsh basei label percentages alliws ola 
- [] Self-supervised FL algorithm implementation k peiramata gia fraction=0.5/1 k classes=10
- [] Give feature: ignore evaluate_fn and ignore results related to test_acc... indexing me rounds na dw gia na xwraw 2-3 client ray processes. na trexw mono apo miniconda cmd xwris VScode
- [] Na anebasw # client processes kai na bgalw evaluate_fn global test gia faster simulations
- [] Plot algorithm comparison and titles based on .yaml files : Algorithm(color) based on train acc or other accs(diff line w/ same color)
- [] Nea polla peiramata me fraction=1 k higher #classes

## Xrysa tips
- [✓] Read papers gia meta-learning/self-supervised me pre-trained pretext tasks kai downstream tasks
- [] reproduce client manager lock seed https://discuss.flower.ai/t/how-do-i-write-a-custom-client-selection-protocol/74/2 alliws 
clientmanager inherit gia change sampler method
- [] Latex : kefalaio 2.1 gia anagnwrish faghtwn(proklhsh, )kai 1.4 sto neo 2.1 isws . 3.1.1 - match 4.1.1 
- [] 



## Better data vis
- [] X y functions pipeline for new dataset
- [] (If i want best train acc save model after server fit_round func)
- [] Heatmaps labels kai client IDs opws NIID bench paper selida 6 figure 4 me confusion matrix grid. Blepe xtra-computing code
- [] Wandb or neptune.ai for loggers with usage of log_dict or tables per iteration
- [] From strategy per round hook and stream data to a server(mongo/Prometheus/telegraf) and real time plot them(Grafana)


## Far away will see...
- [] Compare preprocess
- [] FedBN or [group normalization layers](https://github.com/stevelaskaridis/Federated-Learning-for-Inference-at-Anytime-and-Anywhere/blob/master/models/resnet_v2.py) on ResNet
- [✓] Choice argument for automated clean up in end of scripts of data(.pt etc.) with os, Path and glob
- [] Hard optimize memory track and reduce computation time
- [] Android https://github.com/kivy/python-for-android?tab=readme-ov-file : model optimization, binarization, quantization, approximation

