# echo-stgnet
the official implement of paper "A spatio-temporal graph convolutional network for ultrasound echocardiographic landmark detection"
# 数据存储格式
我们提供的dataset读取按照下列方式存放的数据。
图像：Data/camus_4ch/Image/patient0001/000.png
其中Data是任意根目录，camus_4ch是数据集的名字，Image表示该文件夹下存储图像，patient000X为第X个样本，每个样本中存储10张图像，从000到009。
点：Data/camus_4ch/Points/46pts/adjmatrix_top5.txt
其中Data是任意根目录，camus_4ch是数据集的名字，Points表示该文件夹下存储点和邻接矩阵包括：adjmatrix_topk.txt，这里k为3,5,7,13,21,33,46。另外还有point.txt，存储了密集采样的46个点。
