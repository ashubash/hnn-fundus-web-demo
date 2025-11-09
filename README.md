# hnn-fundus-web-demo
**Live Demo @**: https://hnn-fundus-web-demo.vercel.app/

**Note**: The pipeline from training the teacher model to distilling it down to the student model was done using the RTX 5090 card. If unavailable, you can alternatively setup an environment on Runpod GPU using the RTX 5090 to get the outputs shown in Colab Notebook, and downsize hyperparams to run the training effectively on less compute.

**To Setup**: Make a new **directory: /workspace/** and upload the notebook from the Colab_Notebook_Setup directory, along with the provided **train_split.csv**, **val_split.csv**, and **test_split.csv** files.

**Run each cell in order. This will**:
  *   Fetch the complete dataset that was made from seven distinct sources, including **ODIR-5K**, **Eyepac-Light v2-512**, **RFMID**, **Eyepacs-DEV** Glaucoma images, and large public archives (**Mendeley, multiEyeImages**).
  *   Train teacher model + calibrate conformal quantile
  *   Store logits + prediction sets
  *   Distills into LightHGNN Student model
  *   Exports to 3.83 MB ONNX model
  *   Provides full metrics + classification report on the unseen test data
