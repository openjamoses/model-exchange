# model-exchange
> This repository contains the source-codes and the analysis results of the study we conducted to assess ONNX and CoreML for converting trained DL models. Two popular DL frameworks, Keras and PyTorch, are used to train five widely used DL models on three popular datasets. The trained models are then converted to ONNX and CoreML and transferred to two runtime environments designated for such formats, to be evaluated.
##### What we did?
We investigated:
- the prediction accuracy before and after conversion.
- The performance (time cost and memory consumption) of converted models.  
- the adversarial robustness of converted models to make sure about the robustness of deployed DL-based software.


##### Our observation
- the prediction accuracy of converted models are at the same level of originals.
- The size of models are reduced after conversion, which can result in optimized DL-based software deployment.
- Leveraging the state-of-the-art adversarial attack approaches, converted models are generally assessed robust at the same level of originals. However, obtained results show that CoreML models are more vulnerable to adversarial attacks compared to ONNX.

##### General message
DL developers should be cautious on the deployment of converted models that may:
- perform poorly while switching from one framework to another,
- have challenges in robust deployment, or
- run slowly, leading to poor quality of deployed DL-based software, including DL-based software maintenance tasks, like bug prediction.
