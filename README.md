### Проделанная работа
* Создал на pytorch модель с активацией selu, затем в onnx файле переименовал этот слой в mySelu. После этого получил ожидаемую ошибку в tensorRT.
* Добавил в plugin папку mySeluPlugin, в которой реализовал кастомный слой selu.
* Отредактировал сэмпл sampleOnnxMnistCoordConvAC, просто заменил загружаемую сеть на нужную мне.
### Сборка
Все необходимые библиотеки указаны в оригинальном репозитории TensorRT \
Для сборки из корня репозитория необходимо выполнить команды:
```
mkdir build && cd build && cmake ..
cd samples/opensource/sampleOnnxMnistCoordConvAC/
make
```
В папке build появится бинарник sample_onnx_mnist_coord_conv_ac. Заходим туда и выполняем:\
```LD_LIBRARY_PATH=. ./sample_onnx_mnist_coord_conv_ac -d ../data/```
### Результаты
По итогам выполнения команды выше получаем следующий вывод:
```
&&&& RUNNING TensorRT.sample_onnx_mnist_coord_conv_ac # ./sample_onnx_mnist_coord_conv_ac -d ../data/
[04/14/2021-20:30:12] [I] Building and running a GPU inference engine for Onnx MNIST
[04/14/2021-20:30:12] [I] [TRT] ----------------------------------------------------------------
[04/14/2021-20:30:12] [I] [TRT] Input filename:   ../data/lenet.onnx
[04/14/2021-20:30:12] [I] [TRT] ONNX IR version:  0.0.6
[04/14/2021-20:30:12] [I] [TRT] Opset version:    9
[04/14/2021-20:30:12] [I] [TRT] Producer name:    pytorch
[04/14/2021-20:30:12] [I] [TRT] Producer version: 1.8
[04/14/2021-20:30:12] [I] [TRT] Domain:           
[04/14/2021-20:30:12] [I] [TRT] Model version:    0
[04/14/2021-20:30:12] [I] [TRT] Doc string:       
[04/14/2021-20:30:12] [I] [TRT] ----------------------------------------------------------------
[04/14/2021-20:30:12] [I] [TRT] No importer registered for op: mySelu. Attempting to import as plugin.
[04/14/2021-20:30:12] [I] [TRT] Searching for plugin: mySelu, plugin_version: 1, plugin_namespace: 
[04/14/2021-20:30:12] [I] [TRT] Successfully created plugin: mySelu
[04/14/2021-20:30:12] [I] [TRT] No importer registered for op: mySelu. Attempting to import as plugin.
[04/14/2021-20:30:12] [I] [TRT] Searching for plugin: mySelu, plugin_version: 1, plugin_namespace: 
[04/14/2021-20:30:12] [I] [TRT] Successfully created plugin: mySelu
[04/14/2021-20:30:12] [04/14/2021-20:30:13] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[04/14/2021-20:30:14] [I] [TRT] Detected 1 inputs and 1 output network tensors.
[04/14/2021-20:30:14] [04/14/2021-20:30:14] [04/14/2021-20:30:14] [I] Input:
[04/14/2021-20:30:14] [I] @@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@===--%@@@@@@@
@@@@@@@@@@@@@@*     -@@@@@@@
@@@@@@@@@@@@%:       %@@@@@@
@@@@@@@@@@@@   .=*+  #@@@@@@
@@@@@@@@@@@*   %@@@: -@@@@@@
@@@@@@@@@@#   +@@@@: -@@@@@@
@@@@@@@@@@*  -@@@@@: -@@@@@@
@@@@@@@@@@* .%@@@@@: #@@@@@@
@@@@@@@@@@@=%@@@@@# .@@@@@@@
@@@@@@@@@@@@@@@@@%. -@@@@@@@
@@@@@@@@@@@@@@@@@+  +@@@@@@@
@@@@@@@@@@#%@@@@+  +@@@@@@@@
@@@@@@@@#-  --%+   #@@@@@@@@
@@@@@@@%          .@@@@@@@@@
@@@@@@@-           %@@@@@@@@
@@@@@@#            -#@@@@@@@
@@@@@@-             -@@@@@@@
@@@@@@-       --    -@@@@@@@
@@@@@@-     -*@@%-  +@@@@@@@
@@@@@@%- -=*@@@@@@%*@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

[04/14/2021-20:30:14] [I] Output:
[04/14/2021-20:30:14] [I]  Prob 0  0.0000 Class 0: 
[04/14/2021-20:30:14] [I]  Prob 1  0.0000 Class 1: 
[04/14/2021-20:30:14] [I]  Prob 2  1.0000 Class 2: **********
[04/14/2021-20:30:14] [I]  Prob 3  0.0000 Class 3: 
[04/14/2021-20:30:14] [I]  Prob 4  0.0000 Class 4: 
[04/14/2021-20:30:14] [I]  Prob 5  0.0000 Class 5: 
[04/14/2021-20:30:14] [I]  Prob 6  0.0000 Class 6: 
[04/14/2021-20:30:14] [I]  Prob 7  0.0000 Class 7: 
[04/14/2021-20:30:14] [I]  Prob 8  0.0000 Class 8: 
[04/14/2021-20:30:14] [I]  Prob 9  0.0000 Class 9: 
[04/14/2021-20:30:14] [I] 
&&&& PASSED TensorRT.sample_onnx_mnist_coord_conv_ac # ./sample_onnx_mnist_coord_conv_ac -d ../data/
```
Видим, что плагин успешно использовался и модель отработала корректно.
