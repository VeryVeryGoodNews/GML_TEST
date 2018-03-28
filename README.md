# GML_TEST
This repo contains inception_model.py which is the inceptionv3 model written in keras and saved with intention of deploying to google cloud ml-engine for serving... adapted from Hayato Yoshikawa code here: https://medium.com/google-cloud/serverless-transfer-learning-with-cloud-ml-engine-and-keras-335435f31e15

It also includes a instance.json which is a b64 encoded image for testing the deployed model. Deployment to google cloud ml-engine was done. But upon testing via json request, the following error msg was returned:


Gives error:

<{
  "error": "Prediction failed: Error during model execution: AbortionError(code=StatusCode.NOT_FOUND, details=\"FeedInputs: unable to find feed output input_b64:0\")"
}>

