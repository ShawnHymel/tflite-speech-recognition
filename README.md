TensorFlow Lite Speech Recognition Demo
========

This project is a demonstration on how to use TensorFlow and Keras to train a Convolutional Neural Network (CNN) to recognize the wake word "stop" among other words. In addition, it contains another Python example that uses TensorFlow Lite to run inference on the trained model to recognize the spoken word "stop" on a Raspberry Pi (or other Linux-based computer).

The full articles that explain how these programs work and how to use them can be found here:
%%%TODO: Links%%%

Here are the accompanying videos that explain how to use TensorFlow to train and deploy a speech recognition model:
%%%TODO: Links%%%

Prerequisites
--------------

You will need to install TensorFlow, Keras, and Jupyter Notebook on your desktop or laptop. This guide (%%%TODO: Link%%%) will walk you through that process. 

Alternatively, you can use [Google Colab](https://colab.research.google.com/) to run a Jupyter Notebook instance in the cloud, however, loading files (e.g. training samples) will require you to upload them to Google Drive and write different code to import them into your program. [This guide](https://towardsdatascience.com/3-ways-to-load-csv-files-into-colab-7c14fcbdcb92) offers some tips on how to do that.

Finally, you will need the TensorFlow Lite inference engine installed on your Raspberry Pi (or target deployment device). Instructions to do that can be found here: [https://www.tensorflow.org/lite/guide/python](https://www.tensorflow.org/lite/guide/python).

For hardware, you will need a microphone for your Raspberry Pi to capture audio. This [USB microphone from Adafruit](https://www.adafruit.com/product/3367) worked well for me. If you'd like to see an LED flash whenever you say the word "stop," you can connect an LED to the Raspberry Pi as shown in [this tutorial](https://raspberrypihq.com/making-a-led-blink-using-the-raspberry-pi-and-python/). Follow [these instructions](https://learn.adafruit.com/usb-audio-cards-with-a-raspberry-pi/instructions) to make sure you can record audio on your Raspberry Pi.

Getting Started
---------------

First, download and unzip the [Google Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) on your computer. Since this example uses the Google Speech Commands dataset, I am required (and gratefully so) to give them credit for collecting and releasing this data for everyone to use. To learn more about it (and even contribute samples of your own voice!), please see [this page](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md).

Open **01-speech-commands-mfcc-extraction** in Jupyter Notebook. Change the `dataset_path` variable to point to the location of the unzipped Google Speech Commands dataset directory on your computer. Run the entire script. The script will convert all speech samples (excluding the _background_noide_ set) to their [Mel Frequency Cepstral Coefficients](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) (MFCCs), divide them into training, validation, and test sets, and save them as tensors in a file named `all_targets_mfcc_sets.npz`.

Next, open **02-speech-commands-mfcc-classifier** in Jupyter Notebook and change the `dataset_path` variable to point to the location of the unzipped Google Speech Commands dataset directory. Feel free to change the `wake_word` variable to any other wake word available in the Speech Commands dataset. Run the entire script. It will read the MFCCs from the file made in the first script, build a CNN (credit goes to [this GeeksforGeeks article](https://www.geeksforgeeks.org/python-image-classification-using-keras/) for the CNN structure), and train it using the training features we created (MFCCs). The script will then save the model in the `wake_word_stop_model.h5` file.

Open **03-tflite-model-converter** and make sure that `keras_model_filename` points to the location of the .h5 model we created in the previous script. Run this script to convert the .h5 model into a .tflite model.

Copy the newly created **wake_word_stop_lite.tflite** and **04-rpi-tflite-audio-stream.py** files to the same directory somewhere on your Raspberry Pi. Run the **04-rpi-tflite-audio-stream.py** script. You should see numbers scrolling in the terminal that correspond to the output of the CNN; these are confidence levels that the last 1 second of captured audio contained the word "stop." If you say "stop," the program should print out the word. If you have an LED connected to pin 8, you should see it also flash briefly.

![Output of Raspberry Pi running TensorFlow Lite to classify wake word](https://raw.githubusercontent.com/ShawnHymel/tflite-speech-recognition/master/Images/tflite-pi-wake-word-output.png)

License
-------

All code in this repository is for demonstration purposes and licensed under [Beerware](https://en.wikipedia.org/wiki/Beerware).

Distributed as-is; no warranty is given.