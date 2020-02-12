# MaskRCNN_models_Flask_API
Flask App Web server for MaskRCNN based models.

**Tensorflow** : v1.15.0

**Flask** : v1.1.1

Move your model.h5 file into ./models/ and change the MODEL_PATH in Flaskapp.py. 
* Run and test the API with POSTMAN.
* Note: There is no web interface for the server so all the testing is to be done using POSTMAN only.



<h2> For running over Google colab: </h2>

Install flask-ngrok by running the following in the colab cell:

`!pip install flask-ngrok==0.0.25`
<br>
`!pip install flask==0.12.2`

`from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app) `

and run the app. 

You will get response something like this:

```* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
Re-starting from epoch 30
Loading Weights
Model loaded. Start serving...
 * Running on http://e7e34bd0.ngrok.io
 * Traffic stats available on http://127.0.0.1:4040
 ```
 * Enjoy testing the API on POSTMAN, by using your image as body parameter and changing the file type from 'Text' to 'File' on the adjoining drop down button.
 * Select your file and click 'Send'.
 
