# Multiple Regression Using Pytorch (GPU or CPU)
This is an example of Multiple Regression using Pytorch (GPU or CPU). The data used here is the concrete data and can be downloaded from: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv. The code split the data to training and validation data sets and normalize the feature part, then load it to training and validation loaders. The user has the choice to use the subclass and sequential model. The code will generate loss and accuracy data and plot them. It also plots the validation data with their predictions.
The figure below show the loss and accuracy versus epoch for training and validation process:
![loss_accuracy_epoch](https://user-images.githubusercontent.com/12114448/222917557-0dd908a5-3d85-4156-b028-8784b3bcef95.png)


The figures below show the features versus target for validation data:
![Strength_vs_Age](https://user-images.githubusercontent.com/12114448/222917571-ac3f6c85-8de4-4874-be1d-92c83e7589e6.png)
![Strength_vs_Blast Furnace Slag](https://user-images.githubusercontent.com/12114448/222917572-734361e0-05ce-47f6-a418-2cc6e5ea9b4f.png)
![Strength_vs_Cement](https://user-images.githubusercontent.com/12114448/222917573-307a9f99-ad17-45b5-98dc-4f87b77ef854.png)
![Strength_vs_Coarse Aggregate](https://user-images.githubusercontent.com/12114448/222917574-3a7f51b2-4c3e-4e8b-86e7-9f26ecb7eb12.png)
![Strength_vs_Fine Aggregate](https://user-images.githubusercontent.com/12114448/222917575-f2f32b50-fd28-49b9-9faa-57bf4f77eacb.png)
![Strength_vs_Fly Ash](https://user-images.githubusercontent.com/12114448/222917576-ad1e5363-3889-46d7-86d0-7a7fd6de565a.png)
![Strength_vs_Superplasticizer](https://user-images.githubusercontent.com/12114448/222917577-9b74fda7-2997-4f1a-b53a-2e3640afc5df.png)
![Strength_vs_Water](https://user-images.githubusercontent.com/12114448/222917578-d4045aeb-d2bd-4c7b-8642-747c30c30391.png)
