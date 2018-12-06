# Deep Photo Enhancment: Using Improved 1-Way GAN
Our implmented project used an improved one-way GAN in order to provide an enhanced image. The network learns characteristics from a set of photographs that have the desired features the user prefers and applies these characteristics to the input image, transforming it into an enhanced image. The network is based on a Wasserstein GAN with additional modifications due to the infamous issue of convergence in one-way GANs. The first modification consisted of adding global features to the U-Net in order to capture the overall aesthetics of the image and locally adjusting the image. The second aspect was adding an adaptive weight scheme to reduce the sensitivity of the network on the weighting parameter of the gradient penalty and allowing for a more stable convergence. The last aspect added was individual batch normalization with the idea of improving the loss in the generator by maintaining a consistent set of inputs. With these modifications on the one-way GAN, we compare our results to the approach proposed by Chen using the PSNR scores and an evaluation on the visual quality of the images.


### Code Structure
* ```Analysis Directory```: Provides the parsing of the log results in order to graph and analyze the pre-training and training network.
* ```Code Development Directory```: A vareity of network combinations that were tested in order to further improve the one-way GAN
* ```Results Directory```: Images of our final results
* ```Source Code Directory```: Our final implementation of the one-way GAN
* ```Scripts Directory ```: Scripts to download the images from MIT Adobe-5K Dataset

### Running the Code
* Install the required packages in a virtual or conda environment:
	* Python 3.6
	* Pytorch with GPU
* Create all the necessary directories under ```model/``` directory in order to save the log file with the results and the checkpoints from the network and ```image/``` directory in order to save all the images
* To pre-train the network, you need to run the file ```1WayGAN_PreTrain.py``` by using the following command:
```
python 1WayGAN_PreTrain.py
```
* Once you have pre-trained the network, you will need to upload the last checkpoint into the the ```1WayGAN_Train.py```
* To train the network, you need to run the file ```1WayGAN_Train.py``` by using the following command:
```
python 1WayGAN_Train.py
```
* Once the model runs, your ```model/``` directory will contain all the results from the network.
