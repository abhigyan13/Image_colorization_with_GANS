# Image_colorization_with_GANS

Gray-scale Image Colorization of images using a Conditional Deep Convolutional Generative Adversarial Network (DCGAN). This is a PyTorch implementation of the Conditional DCGAN(Deep Convolutional Generative Adversarial Networks) based on the research paper [Image Colorization using Generative Adversarial Networkshttps](https://arxiv.org/pdf/1803.05400.pdf)

The model is trained on a subset of Places-365 data containing 5000 images . I was not able to train on large dataset due to the non availability of GPU and other hardwares . The suggestions from the paper have been used for hyperparameter turning . The model was trained for about 60 epochs on Google Colab . The model displays some good results on the training set which are shown below . 

Drive Link for Dataset https://drive.google.com/drive/folders/1h_9OlPtBA1tHylqtBItxP9SncVgMM0Zm?usp=sharing

# Network Architecture 
![march](https://user-images.githubusercontent.com/61914611/110186190-76b97280-7e3a-11eb-8d22-a2952047ee88.png)
# Trained Losses

               Generator Loss Plot
![gen](https://user-images.githubusercontent.com/61914611/110186791-57234980-7e3c-11eb-8411-a27825a0466a.png)
 
 
               Discriminator Loss Plot
![disc](https://user-images.githubusercontent.com/61914611/110186789-568ab300-7e3c-11eb-9c5b-7059887930c4.png)
               
# Some Results
----------- Gray Scale Image ------------------------ Original Image----------------------------- Generated Image --------
![i5](https://user-images.githubusercontent.com/61914611/110186455-52aa6100-7e3b-11eb-945b-726311e9fc0e.jpg)
![i3](https://user-images.githubusercontent.com/61914611/110186458-53db8e00-7e3b-11eb-867c-84eddbd62761.jpg)
![i16](https://user-images.githubusercontent.com/61914611/110186460-550cbb00-7e3b-11eb-8122-67a6d4a9932c.jpg)
![i22 (1)](https://user-images.githubusercontent.com/61914611/110186461-55a55180-7e3b-11eb-85cd-df520e8bed8f.jpg)
![i20 (1)](https://user-images.githubusercontent.com/61914611/110186462-56d67e80-7e3b-11eb-879d-a9b14f591846.jpg)
![i17](https://user-images.githubusercontent.com/61914611/110186463-576f1500-7e3b-11eb-9f8d-1f164071ad42.jpg)
